#pragma once
#include <ATen/cuda/CUDAContext.h>
// #include <torch/all.h>
#include <cmath>


// #ifndef USE_ROCM
#include <cub/util_type.cuh>
#include <cub/cub.cuh>
#include "../../utilsq.cuh"
#include "../../utils.cuh"


namespace vllm {


template <typename scalar_t, typename scale_type>
__global__ void dynamic_scaled_int8_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size) {
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;

  // Must be performed using 64-bit math to avoid integer overflow.
  out += token_idx * hidden_size;
  input += token_idx * hidden_size;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(input[i]);
    val = val > zero ? val : -val;
    absmax_val = val > absmax_val ? val : absmax_val;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = block_absmax_val / 127.0f;
  }
  __syncthreads();

  float const tmp_scale = 127.0f / block_absmax_val;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[i] = float_to_int8_rn(static_cast<float>(input[i]) * tmp_scale);
  }
}


}  // namespace vllm

template <typename T>
struct QuantizerScalefloat
{
    int dim;
    int8_t* output;
    // using ScaleType = typename std::conditional<use_per_token_quant, half *, half>::type;
    float* output_scale;
    
    QuantizerScalefloat(int dim)
    {
        this->dim = dim;
    }

    int64_t init_scale_ptr(Memory * memory, int32_t num_tokens, int64_t offset){
        return memory->allocate((void**)&this->output_scale, offset, num_tokens * sizeof(float));
    }

    int64_t init_output_ptr(Memory * memory, int32_t num_tokens, int64_t offset){
        int64_t output_offset =  memory->allocate((void**)&this->output, offset, num_tokens * dim * sizeof(int8_t));
        return this->init_scale_ptr(memory, num_tokens, output_offset);
    }

    
    void invoke(const Stream& stream,
                T *input,
                int num_tokens)
    {
        // 配置线程块和线程网格
        dim3 grid(num_tokens);
        dim3 block(std::min(dim, 1024));

        // 调用核函数
        vllm::dynamic_scaled_int8_quant_kernel<T, float>
              <<<grid, block, 0, stream.stream>>>(
                  input, output,
                  output_scale, dim);

    }
};