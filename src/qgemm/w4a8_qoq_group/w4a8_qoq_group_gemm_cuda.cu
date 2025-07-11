// Implemented by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }

#include <stdio.h> 
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include "../../utils.cuh"
#include "w4a8_qoq_group_gemm_cuda.cuh"

#define OP_M 16
#define OP_N 8
#define OP_K 32
#define INTRIN_M 16
#define INTRIN_N 16
#define INTRIN_K 32
#define WARP_SIZE 32
#define SMEM_PAD_A 0
#define SMEM_PAD_B 0
#define PACK_SIZE 16
#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif
#define KERNEL_LAUNCH_CODE                                                                                   \
  constexpr int NUM_WARPS = (CTA_M / WARP_M) * (CTA_N / WARP_N) * (CTA_K / WARP_K);                          \
  constexpr int SCALES_SMEM_SIZE = (G >= CTA_K) ? (CTA_N * STAGES * 2) : (CTA_N * (CTA_K / G) * STAGES * 2); \
  constexpr int kSmemByteSize =                                                                              \
      ((CTA_M * (CTA_K + SMEM_PAD_A) + CTA_N * (CTA_K + SMEM_PAD_B) / 2) * STAGES + SCALES_SMEM_SIZE) *      \
      sizeof(int8_t);                                                                                        \
  if (kSmemByteSize >= 99 * 1024)                                                                            \
  {                                                                                                          \
    printf("This kernel requires %d Bytes of shared memory, which exceeds "                                  \
           "device limit.\n",                                                                                \
           kSmemByteSize);                                                                                   \
    return ;                                                                                                 \
  }                                                                                                          \
  int num_blocks_m = (num_out_feats + CTA_M - 1) / CTA_M;                                                    \
  int num_blocks_n = num_out_channels / CTA_N / 1;                                                           \
  const int log_tile = get_log_tile<8>((num_out_feats + CTA_M - 1) / CTA_M);                                 \
  const int tile_shift = 1 << log_tile;                                                                      \
  dim3 num_blocks(num_blocks_n *tile_shift,                                                                  \
                  (num_blocks_m + tile_shift - 1) / tile_shift);                                             \
  dim3 threads_per_block(WARP_SIZE, NUM_WARPS);                                                              \
  auto kernel_func =                                                                                         \
      dense_kernel0<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, STAGES, G>;                                 \
  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize,                             \
                       kSmemByteSize);                                                                       \
  kernel_func<<<num_blocks, threads_per_block, kSmemByteSize, stream.stream>>>(                                             \
      in_feats, kernel, zeros, scales_i8, wscales, ascales, out_feats, num_in_feats, num_out_channels,       \
      num_in_channels);

template <int N>
__inline__ __host__ __device__ int get_log_tile(int n)
{
  if (N >= 8 && n >= 6)
    return 3;
  else if (N >= 4 && n >= 3)
    return 2;
  else if (N >= 2 && n >= 2)
    return 1;
  else
    return 0;
}

__inline__ __device__ uint2 get_block_idx_mapping(int blockIdx_x,
                                                  int blockIdx_y,
                                                  int log_tile)
{
  return make_uint2((blockIdx_x >> log_tile),
                    (blockIdx_y << log_tile) +
                        ((blockIdx_x) & ((1 << (log_tile)) - 1)));
}

__inline__ __device__ uint32_t cast_smem_ptr_to_uint(void const *const ptr)
{
  uint32_t smem_int_ptr;

  asm("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, "
      "smem_ptr; }\n"
      : "=r"(smem_int_ptr)
      : "l"(ptr));

  return smem_int_ptr;
}

__inline__ __device__ void ldmatrix_m8n8_x4_b16(int8_t *shared_warp, int ax0_0,
                                                uint32_t addr)
{
  __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x4.shared.b16"
                       "{%0, %1, %2, %3}, [%4];"
                       : "=r"(((unsigned *)(shared_warp + (ax0_0 * 16)))[0]),
                         "=r"(((unsigned *)(shared_warp + (ax0_0 * 16)))[1]),
                         "=r"(((unsigned *)(shared_warp + (ax0_0 * 16)))[2]),
                         "=r"(((unsigned *)(shared_warp + (ax0_0 * 16)))[3])
                       : "r"(addr));
}

__inline__ __device__ void
ldmatrix_m8n8_x4_trans_b16(int8_t *shared_warp, int ax0_0, uint32_t addr)
{
  __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
                       "{%0, %1, %2, %3}, [%4];"
                       : "=r"(((unsigned *)(shared_warp + (ax0_0 * 16)))[0]),
                         "=r"(((unsigned *)(shared_warp + (ax0_0 * 16)))[1]),
                         "=r"(((unsigned *)(shared_warp + (ax0_0 * 16)))[2]),
                         "=r"(((unsigned *)(shared_warp + (ax0_0 * 16)))[3])
                       : "r"(addr));
}

// function from lmdeploy
__inline__ __device__ void
cp_async_cg_A(uint32_t smem_int_ptr, const uint4 *__restrict__ src, bool mask)
{
  const int cp_size = 16;
  asm volatile("{"
                "  .reg .pred p;"
                "  setp.ne.b32 p, %0, 0;"
                "  @p cp.async.cg.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;"
                "}" ::"r"((int)mask),
                "r"(smem_int_ptr),
                "l"(src),
                "n"(cp_size));
}

__device__ __inline__ void mma_m16n8k32(void *C_warp, void *A_shared_warp,
                                        void *B_shared_warp)
{
  __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
      : "=r"(((int *)C_warp)[0]), "=r"(((int *)C_warp)[1]),
        "=r"(((int *)C_warp)[2]), "=r"(((int *)C_warp)[3])
      : "r"(((unsigned *)A_shared_warp)[0]),
        "r"(((unsigned *)A_shared_warp)[1]),
        "r"(((unsigned *)A_shared_warp)[2]),
        "r"(((unsigned *)A_shared_warp)[3]),
        "r"(((unsigned *)B_shared_warp)[0]),
        "r"(((unsigned *)B_shared_warp)[1]), "r"(((int *)C_warp)[0]),
        "r"(((int *)C_warp)[1]), "r"(((int *)C_warp)[2]),
        "r"(((int *)C_warp)[3]));
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int SHARED_K_ITERS,
          int STAGES>
__device__ __inline__ void
global_to_share_one_stage_A(int8_t *src, int8_t *dst, int global_ncols,
                            int cta_offset_m, int cta_offset_n,
                            int global_iter_k, int shared_iter_k, bool mask,
                            bool *preds)
{
  constexpr int total_global_iters = (CTA_M * CTA_K) / PACK_SIZE / CTA_SIZE;
  constexpr int partial_global_iters = total_global_iters / SHARED_K_ITERS;
  constexpr int cta_step_m_or_n = (CTA_SIZE * PACK_SIZE) / CTA_K;
  constexpr int warp_step_m_or_n = (WARP_SIZE * PACK_SIZE) / CTA_K;
  constexpr int threads_per_row = CTA_K / PACK_SIZE;
  constexpr int kSmemCol = CTA_K + SMEM_PAD_A;
  int8_t *dst_hoisted = dst;
  int8_t *src_hoisted = src + global_iter_k * CTA_K;

  if (mask)
  {
#pragma unroll
    for (int _global_iter = 0; _global_iter < partial_global_iters;
         ++_global_iter)
    {
      int global_iter = shared_iter_k * partial_global_iters + _global_iter;
      void *dst_ptr =
          (void *)(dst_hoisted + global_iter * cta_step_m_or_n * kSmemCol);
      uint4 *src_ptr =
          (uint4 *)(src_hoisted + global_iter * cta_step_m_or_n * global_ncols);
      if constexpr (STAGES > 1)
      {
        uint32_t addr = cast_smem_ptr_to_uint(dst_ptr);
        cp_async_cg_A(addr, src_ptr, preds[global_iter]);
      }
      else
      {
        if (preds[global_iter])
          *(uint4 *)dst_ptr = *src_ptr;
      }
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int SHARED_K_ITERS,
          int STAGES>
__device__ __inline__ void
global_to_share_one_stage_B(int8_t *src, int8_t *dst, int global_ncols,
                            int cta_offset_m, int cta_offset_n,
                            int global_iter_k, int shared_iter_k, bool mask)
{
  constexpr int total_global_iters = (CTA_N * CTA_K) / 32 / CTA_SIZE;
  constexpr int NUM_WARPS = CTA_SIZE / WARP_SIZE;
  constexpr int warps_per_row = CTA_K / 32;
  constexpr int cta_step_m_or_n = NUM_WARPS / warps_per_row;
  constexpr int kSmemCol = CTA_K;
  int8_t *dst_hoisted = dst;
  int8_t *src_hoisted = src + global_iter_k * CTA_K * PACK_SIZE;

#pragma unroll
  for (int global_iter = 0; global_iter < total_global_iters; ++global_iter)
  {
    void *dst_ptr = (void *)(dst_hoisted + global_iter * cta_step_m_or_n *
                                               kSmemCol * PACK_SIZE);
    uint4 *src_ptr = (uint4 *)(src_hoisted + global_iter * cta_step_m_or_n *
                                                 global_ncols * PACK_SIZE);
    if constexpr (STAGES > 1)
    {
      uint32_t addr = cast_smem_ptr_to_uint(dst_ptr);
      cp_async_cg_A(addr, src_ptr, mask);
    }
    else
    {
      if (mask)
        *(uint4 *)dst_ptr = *src_ptr;
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int STAGES, int G>
__device__ __inline__ void global_to_share_one_stage_zeros(int8_t *src, int8_t *dst, int global_ncols, int cta_offset_m, int cta_offset_n, int global_iter_k, int shared_iter_k, bool mask)
{
  constexpr int threads_needed = CTA_N / PACK_SIZE / 1;
  constexpr int threads_used = threads_needed < CTA_SIZE ? threads_needed : CTA_SIZE;
  constexpr int total_global_iters = CTA_N / PACK_SIZE / threads_used;
  constexpr int threads_per_row = CTA_N / PACK_SIZE;
  constexpr int kSmemCol = CTA_N;
  bool local_mask = mask & (threadIdx.y * WARP_SIZE + threadIdx.x < threads_used);
  int g_idx = global_iter_k * CTA_K / G;

  void *dst_ptr = (void *)(dst + (threadIdx.x % threads_per_row) * PACK_SIZE);
  uint4 *src_ptr = (uint4 *)(src + g_idx * global_ncols + cta_offset_n + (threadIdx.x % threads_per_row) * PACK_SIZE);
  if (STAGES > 1)
  {
    uint32_t addr = cast_smem_ptr_to_uint(dst_ptr);
    cp_async_cg_A(addr, src_ptr, local_mask);
  }
  else
  {
    if (local_mask)
    {
      *(uint4 *)dst_ptr = *src_ptr;
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int STAGES>
__device__ __inline__ void
share_to_reg_one_stage_A(int8_t *src, int8_t *dst, int warp_offset_m,
                         int warp_offset_n, int k_0_1, int shared_iters)
{
  constexpr int kSmemCol = CTA_K + SMEM_PAD_A;
  int ld_col = (k_0_1 * INTRIN_K + (threadIdx.x / 16) * 16) / PACK_SIZE;

  for (int shared_iter = 0; shared_iter < shared_iters; ++shared_iter)
  {
    int ld_row = warp_offset_m + shared_iter * INTRIN_M + (threadIdx.x % 16);
    int ld_col_swizzled = ld_col ^ (ld_row / 2) & 3;
    void *addr_ptr =
        (void *)(src + ld_row * kSmemCol + ld_col_swizzled * PACK_SIZE);
    uint32_t addr = cast_smem_ptr_to_uint(addr_ptr);
    ldmatrix_m8n8_x4_b16(dst, shared_iter, addr);
  }
}

template <int WARP_K, int CTA_N, int CTA_K, int CTA_SIZE, int STAGES, int G>
__device__ __inline__ void
share_to_reg_one_stage_B(int8_t *src, int8_t *dst, int8_t *zeros, int8_t *scales_i8,
                         int warp_offset_m, int warp_offset_n, int k_0_0,
                         int k_0_1, int shared_iters)
{
  constexpr int kSmemCol = CTA_K + SMEM_PAD_B;
#pragma unroll
  for (int shared_iter = 0; shared_iter < shared_iters; ++shared_iter)
  {
    uint4 loaded =
        *((uint4 *)(src) + warp_offset_n / 32 * kSmemCol +
          shared_iter * 32 / 32 * kSmemCol + k_0_1 * INTRIN_K + threadIdx.x);
    uint32_t loaded_0 = loaded.x & 0x0F0F0F0F;
    uint32_t loaded_4 = (loaded.x & 0xF0F0F0F0) >> 4;
    uint32_t loaded_2 = loaded.y & 0x0F0F0F0F;
    uint32_t loaded_6 = (loaded.y & 0xF0F0F0F0) >> 4;
    uint32_t loaded_1 = loaded.z & 0x0F0F0F0F;
    uint32_t loaded_5 = (loaded.z & 0xF0F0F0F0) >> 4;
    uint32_t loaded_3 = loaded.w & 0x0F0F0F0F;
    uint32_t loaded_7 = (loaded.w & 0xF0F0F0F0) >> 4;

    auto ptr = (uint32_t *)dst + shared_iter * 8;
    int scales_zeros_offset = warp_offset_n + (threadIdx.x / 4) * 4 + shared_iter * 32;
    uint32_t packed_scales = *reinterpret_cast<uint32_t *>(scales_i8 + scales_zeros_offset);
    uint32_t packed_zeros = *reinterpret_cast<uint32_t *>(zeros + scales_zeros_offset);

    uint32_t scale_0 = packed_scales & 0xFF;
    uint32_t zero_point_0 = __byte_perm(packed_zeros, 0, 0x00000000);
    uint32_t ptr_0 = loaded_0 * scale_0;
    uint32_t ptr_1 = loaded_1 * scale_0;
    ptr[0] = __vadd4(ptr_0, zero_point_0);
    ptr[1] = __vadd4(ptr_1, zero_point_0);

    uint32_t scale_1 = (packed_scales & 0xFF00) >> 8;
    uint32_t zero_point_1 = __byte_perm(packed_zeros, 0, 0x00001111);
    uint32_t ptr_2 = loaded_2 * scale_1;
    uint32_t ptr_3 = loaded_3 * scale_1;
    ptr[2] = __vadd4(ptr_2, zero_point_1);
    ptr[3] = __vadd4(ptr_3, zero_point_1);

    uint32_t scale_2 = (packed_scales & 0xFF0000) >> 16;
    uint32_t zero_point_2 = __byte_perm(packed_zeros, 0, 0x00002222);
    uint32_t ptr_4 = loaded_4 * scale_2;
    uint32_t ptr_5 = loaded_5 * scale_2;
    ptr[4] = __vadd4(ptr_4, zero_point_2);
    ptr[5] = __vadd4(ptr_5, zero_point_2);

    uint32_t scale_3 = (packed_scales & 0xFF000000) >> 24;
    uint32_t zero_point_3 = __byte_perm(packed_zeros, 0, 0x00003333);
    uint32_t ptr_6 = loaded_6 * scale_3;
    uint32_t ptr_7 = loaded_7 * scale_3;
    ptr[6] = __vadd4(ptr_6, zero_point_3);
    ptr[7] = __vadd4(ptr_7, zero_point_3);
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int WARP_M, int WARP_N, int WARP_K,
          int STAGES, int G>
__global__ void dense_kernel0(int8_t *__restrict__ A, int8_t *__restrict__ B,
                              int8_t *__restrict__ zeros, int8_t *__restrict__ scales_i8,
                              half2 *__restrict__ wscales, half *__restrict__ ascales,
                              half *__restrict__ C, int M, int64_t N, int64_t K)
{
  constexpr int SPLITK = 1;
  constexpr int NUM_WARPS_MN = CTA_M / WARP_M * CTA_N / WARP_N;
  constexpr int NUM_WARPS = NUM_WARPS_MN * CTA_K / WARP_K;
  constexpr int CTA_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int CTA_SIZE_MN = NUM_WARPS_MN * WARP_SIZE;
  constexpr int SLICES = CTA_K / WARP_K;
  int num_blocks_n = (N + CTA_N - 1) / CTA_N;
  int num_blocks_m = (M + CTA_M - 1) / CTA_M;

  int blockIdx_n = blockIdx.x;
  int blockIdx_m = blockIdx.y;
  const int log_tile = get_log_tile<8>((M + CTA_M - 1) / CTA_M);
  const uint2 block_idx_mapping =
      get_block_idx_mapping(blockIdx_n, blockIdx_m, log_tile);
  blockIdx_n = block_idx_mapping.x;
  blockIdx_m = block_idx_mapping.y;

  int C_warp[CTA_M * CTA_N / CTA_SIZE_MN];
  constexpr int kSmemPadKA = CTA_K + SMEM_PAD_A;
  constexpr int kSmemPadKB = CTA_K + SMEM_PAD_B;
  constexpr int kSmemSizeAPerStage = CTA_M * kSmemPadKA;
  constexpr int kSmemSizeBPerStage = CTA_N * kSmemPadKB / 2;
  constexpr int kSmemSizeA = kSmemSizeAPerStage * STAGES;
  constexpr int kSmemSizeB = kSmemSizeBPerStage * STAGES;

  constexpr int scales_load_interval = G >= CTA_K ? G / CTA_K : 1;
  constexpr int scales_per_load = G < CTA_K ? CTA_K / G : 1;
  constexpr int kSmemSizeScales = CTA_N * STAGES;

  extern __shared__ int8_t mem_shared[];
  int8_t *A_shared = mem_shared;

  int8_t *B_shared = mem_shared + kSmemSizeA;
  int8_t *zeros_shared = mem_shared + kSmemSizeA + kSmemSizeB;
  int8_t *scales_i8_shared = mem_shared + kSmemSizeA + kSmemSizeB + kSmemSizeScales;

  int8_t A_shared_warp_[2][WARP_M * WARP_K /
                           WARP_SIZE]; 
  int8_t B_shared_warp_[2][WARP_N * WARP_K /
                           WARP_SIZE]; 
  constexpr int A_total_global_iters = (CTA_M * CTA_K) / PACK_SIZE / CTA_SIZE;
  constexpr int B_total_global_iters = (CTA_N * CTA_K) / PACK_SIZE / CTA_SIZE;
  constexpr int A_src_step_m = (CTA_SIZE * PACK_SIZE) / CTA_K;
  constexpr int A_warp_step_m = (WARP_SIZE * PACK_SIZE) / CTA_K;
  constexpr int A_threads_per_row = CTA_K / PACK_SIZE;

  constexpr int B_warps_per_row = CTA_K / 32;
  constexpr int B_src_step_n = NUM_WARPS / B_warps_per_row;

  int cta_offset_m = blockIdx_m * CTA_M;
  int cta_offset_n = blockIdx_n * CTA_N;
  int warp_mn = threadIdx.y % NUM_WARPS_MN;
  int slice_id = threadIdx.y / NUM_WARPS_MN;
  int warp_offset_m = (warp_mn % (CTA_M / WARP_M)) * WARP_M;
  int warp_offset_n = (warp_mn / (CTA_M / WARP_M)) * WARP_N;
  int warp_offset_k = slice_id * WARP_K;

  for (int i = 0; i < CTA_M * CTA_N / CTA_SIZE_MN; i++)
    C_warp[i] = 0;

  int gemm_iters = (K + CTA_K - 1) / CTA_K;

  int k_0_0_ld = 0;
  int k_0_0 = 0;
  constexpr int prologue_stages = STAGES == 1 ? 1 : STAGES - 1;
  int A_hoisted_row = threadIdx.y * A_warp_step_m + (threadIdx.x / A_threads_per_row);
  int A_hoisted_col = (threadIdx.x % A_threads_per_row);
  int A_hoisted_col_swizzled = A_hoisted_col ^ (A_hoisted_row / 2) & 3;

  int8_t *A_shared_hoisted = A_shared +
                             A_hoisted_row * kSmemPadKA +
                             A_hoisted_col_swizzled * PACK_SIZE;
  int8_t *B_shared_hoisted =
      B_shared + (threadIdx.y % B_warps_per_row) * 32 * PACK_SIZE +
      (threadIdx.y / B_warps_per_row) * kSmemPadKB * PACK_SIZE +
      threadIdx.x * PACK_SIZE;
  int8_t *A_hoisted = A + cta_offset_m * K + A_hoisted_row * K +
                      A_hoisted_col * PACK_SIZE;
  int8_t *B_hoisted = B + cta_offset_n / 32 * K * PACK_SIZE +
                      (threadIdx.y % B_warps_per_row) * 32 * PACK_SIZE +
                      (threadIdx.y / B_warps_per_row) * K * PACK_SIZE +
                      threadIdx.x * PACK_SIZE;

  bool A_g2s_preds[A_total_global_iters];
#pragma unroll
  for (int i = 0; i < A_total_global_iters; i++)
  {
    A_g2s_preds[i] = (cta_offset_m + A_hoisted_row + i * A_src_step_m) < M;
  }

  int *C_shared = reinterpret_cast<int *>(mem_shared);

#pragma unroll
  for (k_0_0_ld = 0; k_0_0_ld < prologue_stages; ++k_0_0_ld)
  {
    global_to_share_one_stage_A<CTA_M, CTA_N, CTA_K, CTA_SIZE, 1, STAGES>(
        A_hoisted, A_shared_hoisted + k_0_0_ld * kSmemSizeAPerStage, K,
        cta_offset_m, cta_offset_n, k_0_0_ld, 0, true, A_g2s_preds);
    global_to_share_one_stage_B<CTA_M, CTA_N, CTA_K, CTA_SIZE, 1, STAGES>(
        B_hoisted, B_shared_hoisted + k_0_0_ld * kSmemSizeBPerStage, K,
        cta_offset_m, cta_offset_n, k_0_0_ld, 0, true);
    global_to_share_one_stage_zeros<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES, G>(
        zeros, zeros_shared + (k_0_0_ld)*CTA_N,
        N, cta_offset_m, cta_offset_n, k_0_0_ld, 0, k_0_0_ld < gemm_iters);
    global_to_share_one_stage_zeros<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES, G>(
        scales_i8, scales_i8_shared + (k_0_0_ld)*CTA_N,
        N, cta_offset_m, cta_offset_n, k_0_0_ld, 0, k_0_0_ld < gemm_iters);

    if constexpr (STAGES > 1)
      __pipeline_commit();
  }
  if constexpr (STAGES > 1)
    __pipeline_wait_prior(STAGES - 2);
  __syncthreads();

  share_to_reg_one_stage_A<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES>(
      A_shared + warp_offset_k, A_shared_warp_[0], warp_offset_m, warp_offset_n, 0,
      WARP_M / INTRIN_M);
  share_to_reg_one_stage_B<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES, G>(
      B_shared + warp_offset_k * PACK_SIZE, B_shared_warp_[0], zeros_shared, scales_i8_shared,
      warp_offset_m, warp_offset_n, 0, 0, WARP_N / 32);
  constexpr int SHARED_K_ITERS = WARP_K / INTRIN_K;

  for (; k_0_0 < gemm_iters; ++k_0_0, ++k_0_0_ld)
  {
    int ld_stage = k_0_0_ld % STAGES;
    int compute_stage = k_0_0 % STAGES;
    int8_t *A_shared_this_compute_stage;
    int8_t *B_shared_this_compute_stage;
    int8_t *zeros_shared_this_compute_stage;
    int8_t *scales_i8_shared_this_compute_stage;

    for (int iter_k = 0; iter_k < SHARED_K_ITERS; ++iter_k)
    {
      A_shared_this_compute_stage =
          A_shared + compute_stage * kSmemSizeAPerStage + warp_offset_k;
      B_shared_this_compute_stage =
          B_shared + compute_stage * kSmemSizeBPerStage + warp_offset_k * PACK_SIZE;
      zeros_shared_this_compute_stage = zeros_shared + (compute_stage)*CTA_N;
      scales_i8_shared_this_compute_stage = scales_i8_shared + (compute_stage)*CTA_N;

      share_to_reg_one_stage_A<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES>(
          A_shared_this_compute_stage, A_shared_warp_[(iter_k + 1) % 2],
          warp_offset_m, warp_offset_n, (iter_k + 1) % SHARED_K_ITERS,
          WARP_M / INTRIN_M);
      share_to_reg_one_stage_B<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES, G>(
          B_shared_this_compute_stage, B_shared_warp_[(iter_k + 1) % 2],
          zeros_shared_this_compute_stage, scales_i8_shared_this_compute_stage,
          warp_offset_m, warp_offset_n, k_0_0 + (iter_k == SHARED_K_ITERS - 1),
          (iter_k + 1) % SHARED_K_ITERS, WARP_N / 32);
      int8_t *A_shared_warp = A_shared_warp_[iter_k % 2];
      int8_t *B_shared_warp = B_shared_warp_[iter_k % 2];

      for (int j_0_4 = 0; j_0_4 < WARP_N / INTRIN_N; ++j_0_4)
      {
        for (int i_0_3 = 0; i_0_3 < WARP_M / INTRIN_M; ++i_0_3)
        {
          mma_m16n8k32(
              (void *)(C_warp + i_0_3 * WARP_N / INTRIN_N * 8 + j_0_4 * 8),
              (void *)(A_shared_warp + i_0_3 * 16),
              (void *)(B_shared_warp + j_0_4 * 16));
          mma_m16n8k32(
              (void *)(C_warp + i_0_3 * WARP_N / INTRIN_N * 8 + j_0_4 * 8 + 4),
              (void *)(A_shared_warp + i_0_3 * 16),
              (void *)(B_shared_warp + j_0_4 * 16 + 8));
        }
      }

      if (iter_k < SHARED_K_ITERS - 1)
      {
        if constexpr (STAGES == 1)
          __syncthreads();
        global_to_share_one_stage_A<CTA_M, CTA_N, CTA_K, CTA_SIZE,
                                    WARP_K / INTRIN_K, STAGES>(
            A_hoisted, A_shared_hoisted + ld_stage * kSmemSizeAPerStage, K,
            cta_offset_m, cta_offset_n, k_0_0_ld, iter_k,
            k_0_0_ld < gemm_iters, A_g2s_preds);
        global_to_share_one_stage_B<CTA_M, CTA_N, CTA_K, CTA_SIZE,
                                    WARP_K / INTRIN_K, STAGES>(
            B_hoisted, B_shared_hoisted + ld_stage * kSmemSizeBPerStage, K,
            cta_offset_m, cta_offset_n, k_0_0_ld, iter_k,
            k_0_0_ld < gemm_iters);
      }

      if (iter_k == SHARED_K_ITERS - 2)
      {
        if constexpr (STAGES == 1 && SHARED_K_ITERS > 2)
        {
          __syncthreads();
        }
        global_to_share_one_stage_A<CTA_M, CTA_N, CTA_K, CTA_SIZE,
                                    WARP_K / INTRIN_K, STAGES>(
            A_hoisted, A_shared_hoisted + ld_stage * kSmemSizeAPerStage, K,
            cta_offset_m, cta_offset_n, k_0_0_ld, iter_k + 1,
            k_0_0_ld < gemm_iters, A_g2s_preds);
        global_to_share_one_stage_B<CTA_M, CTA_N, CTA_K, CTA_SIZE,
                                    WARP_K / INTRIN_K, STAGES>(
            B_hoisted, B_shared_hoisted + ld_stage * kSmemSizeBPerStage, K,
            cta_offset_m, cta_offset_n, k_0_0_ld, iter_k + 1,
            k_0_0_ld < gemm_iters);
        global_to_share_one_stage_zeros<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES, G>(
            zeros, zeros_shared + (ld_stage)*CTA_N,
            N, cta_offset_m, cta_offset_n, k_0_0_ld, iter_k, k_0_0_ld < gemm_iters);
        global_to_share_one_stage_zeros<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES, G>(
            scales_i8, scales_i8_shared + (ld_stage)*CTA_N,
            N, cta_offset_m, cta_offset_n, k_0_0_ld, iter_k, k_0_0_ld < gemm_iters);
        if constexpr (STAGES > 1)
        {
          __pipeline_commit();
          __pipeline_wait_prior(STAGES - 2);
        }
        compute_stage = (k_0_0 + 1) % STAGES;
        __syncthreads();
      }
    }
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  if constexpr (SLICES > 1)
  {
#pragma unroll
    for (int z = 0; z < SLICES; ++z)
    {
      if (slice_id == z)
      {
#pragma unroll
        for (int ax0_0_1 = 0; ax0_0_1 < WARP_M / INTRIN_M; ++ax0_0_1)
        {
#pragma unroll
          for (int ax1_0_1 = 0; ax1_0_1 < WARP_N / INTRIN_N; ++ax1_0_1)
          {
#pragma unroll
            for (int local_id = 0; local_id < OP_M * 16 / WARP_SIZE; ++local_id)
            {
              if (z > 0)
              {
                C_warp[ax0_0_1 * WARP_N / INTRIN_N * 8 + ax1_0_1 * 8 + local_id] += C_shared[warp_offset_m * CTA_N + ax0_0_1 * OP_M * CTA_N + warp_offset_n + ax1_0_1 * 16 + ((local_id % 4) / 2 * 8 + (threadIdx.x / 4)) * CTA_N + (local_id / 4) * 8 + (local_id % 2) + (threadIdx.x % 4) * 2];
              }
              C_shared[warp_offset_m * CTA_N + ax0_0_1 * OP_M * CTA_N + warp_offset_n + ax1_0_1 * 16 + ((local_id % 4) / 2 * 8 + (threadIdx.x / 4)) * CTA_N + (local_id / 4) * 8 + (local_id % 2) + (threadIdx.x % 4) * 2] = C_warp[ax0_0_1 * WARP_N / INTRIN_N * 8 + ax1_0_1 * 8 + local_id];
            };
          }
        }
      }
      __syncthreads();
    }
    if (slice_id == 0)
    {
#pragma unroll
      for (int ax0_0_1 = 0; ax0_0_1 < WARP_M / INTRIN_M; ++ax0_0_1)
      {
#pragma unroll
        for (int ax1_0_1 = 0; ax1_0_1 < WARP_N / INTRIN_N; ++ax1_0_1)
        {
#pragma unroll
          for (int local_id = 0; local_id < OP_M * 16 / WARP_SIZE; ++local_id)
          {
            C_warp[ax0_0_1 * WARP_N / INTRIN_N * 8 + ax1_0_1 * 8 + local_id] = C_shared[warp_offset_m * CTA_N + ax0_0_1 * OP_M * CTA_N + warp_offset_n + ax1_0_1 * 16 + ((local_id % 4) / 2 * 8 + (threadIdx.x / 4)) * CTA_N + (local_id / 4) * 8 + (local_id % 2) + (threadIdx.x % 4) * 2];
          };
        }
      }
    }
  }

  int row_wb_thd = cta_offset_m + warp_offset_m + (threadIdx.x / 4);
  int col_wb_thd = cta_offset_n + warp_offset_n + (threadIdx.x % 4) * 2;
  if (slice_id == 0)
  {
    for (int ax0_0_1 = 0; ax0_0_1 < WARP_M / INTRIN_M; ++ax0_0_1)
    {
      int row_wb_1 = row_wb_thd + ax0_0_1 * OP_M;
      for (int ax1_0_1 = 0; ax1_0_1 < WARP_N / INTRIN_N; ++ax1_0_1)
      {
        int col_wb_1 = col_wb_thd + ax1_0_1 * 16;
        int *C_warp_local = C_warp + ax0_0_1 * WARP_N / INTRIN_N * 8 + ax1_0_1 * 8;
        for (int local_id = 0; local_id < OP_M * 16 / WARP_SIZE; local_id += 2)
        {
          int row_wb = row_wb_1 + (local_id % 4) / 2 * 8;
          if (row_wb < M)
          {
            int col_wb = col_wb_1 + (local_id / 4) * 8 + (local_id % 2);
            float2 wscale = 	__half22float2(*(wscales + col_wb / 2));
            float ascale = __half2float(ascales[row_wb]);
            float2 psums = make_float2(__int2float_rn(C_warp_local[local_id]), __int2float_rn(C_warp_local[local_id + 1]));
            psums.x *= wscale.x * ascale;
            psums.y *= wscale.y * ascale;
            *reinterpret_cast<half2 *>(C + row_wb * N + col_wb) = __float22half2_rn(psums);
          }
        };
      }
    }
  }
}

void w4a8_qoq_group_gemm_forward_cuda(const Stream& stream,
                        int8_t* in_feats,
                        int8_t* kernel,
                        int8_t* zeros,
                        int8_t* scales_i8,
                        half2* wscales,
                        half* ascales,
                        half* out_feats,
                        int num_in_feats,
                        int num_in_channels,
                        int num_out_feats,
                        int num_out_channels)
{

  constexpr int G = 128;

  if (num_out_feats > 128)
  {
    constexpr int CTA_M = 128;
    constexpr int CTA_N = 128;
    constexpr int CTA_K = 64;
    constexpr int WARP_M = 64;
    constexpr int WARP_N = 32;
    constexpr int WARP_K = 64;
    constexpr int STAGES = 3;
    KERNEL_LAUNCH_CODE
  }
  else if (num_out_feats >= 128)
  {
    if (num_in_channels <= 4096)
    {
      constexpr int CTA_M = 64;
      constexpr int CTA_N = 64;
      constexpr int CTA_K = 64;
      constexpr int WARP_M = 32;
      constexpr int WARP_N = 32;
      constexpr int WARP_K = 64;
      constexpr int STAGES = 4; 
      KERNEL_LAUNCH_CODE
    }
    else
    {
      constexpr int CTA_M = 64;
      constexpr int CTA_N = 64;
      constexpr int CTA_K = 128;
      constexpr int WARP_M = 32;
      constexpr int WARP_N = 32;
      constexpr int WARP_K = 64;
      constexpr int STAGES = 3; 
      KERNEL_LAUNCH_CODE
    }
  }
  else
  {
    constexpr int CTA_M = 32;
    constexpr int CTA_N = 64;
    constexpr int CTA_K = 128;
    constexpr int WARP_M = 32;
    constexpr int WARP_N = 32;
    constexpr int WARP_K = 64;
    constexpr int STAGES = 3;
    KERNEL_LAUNCH_CODE
  }
  return ;
}
