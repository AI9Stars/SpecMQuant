#pragma once
#include <cuda_fp16.h>
#include <cassert>
#include "../../reduction_utils.cuh"
#include "../../utils.cuh"
#include "../../utilsq.cuh"


namespace{
template <typename T, typename scale_type, bool use_per_token_quant>
__global__ void quant_kernel(T *__restrict__ input,
                             int8_t *__restrict__ output, scale_type scale,
                             int num_tokens, int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;

  if constexpr (use_per_token_quant) {
    float amax_val = 0.0f;
    const float zero = 0.0f;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float val = (float)input[token_idx * hidden_size + i];
      val = val > zero ? val : -val;
      if (val > amax_val)
        amax_val = val;
    }

    __shared__ float s_amax;
    const float block_amax_val = blockReduceMax(amax_val);
    if (tid == 0) {
      s_amax = block_amax_val;
      scale[token_idx] = __float2half_rn(block_amax_val / 127.0f);
    }
    __syncthreads();

    float tmp_scale = 127.0f / s_amax;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) * tmp_scale);
    }
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) / __half2float(scale));
    }
  }
}

template <typename T, typename scale_type, bool use_per_token_quant>
__global__ void quant_kernel_fuse_sum(T *__restrict__ input,
                             int8_t *__restrict__ output, 
                             scale_type input_sum, 
                             scale_type scale,
                             int num_tokens, int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;
  const int64_t token_idx_mul_hidden_size = token_idx * int64_t(hidden_size);

  if constexpr (use_per_token_quant) {
    float amax_val = 0.0f;
    float sum_val = 0.0f;
    const float zero = 0.0f;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float val = (float)input[token_idx_mul_hidden_size + i];
      sum_val += val;
      val = val > zero ? val : -val;
      if (val > amax_val)
        amax_val = val;
    }

    __shared__ float s_amax;
    const float block_amax_val = blockReduceMax(amax_val);
    const float block_sum_val = blockReduceSum(sum_val);
    if (tid == 0) {
      s_amax = block_amax_val;
      scale[token_idx] = __float2half_rn(block_amax_val / 127.0f);
      input_sum[token_idx] = __float2half_rn(block_sum_val);
    }
    __syncthreads();

    float tmp_scale = 127.0f / s_amax;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx_mul_hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx_mul_hidden_size + i]) * tmp_scale);
    }
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx_mul_hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx_mul_hidden_size + i]) / __half2float(scale));
    }
  }
}

}

template <typename T, bool use_per_token_quant>
struct Quantizer
{
    int dim;
    int8_t* output;
    half* output_scale;
    
    Quantizer(int dim)
    {
        this->dim = dim;
    }

    int64_t init_output_ptr(Memory * memory, int32_t num_tokens, int64_t offset){
        return memory->allocate((void**)&this->output, offset, num_tokens * dim * sizeof(int8_t));
    }

    int64_t init_output_scale_ptr(Memory * memory, int32_t num_tokens, int64_t offset){
        return memory->allocate((void**)&this->output_scale, offset, num_tokens * sizeof(half));
    }
    

    void invoke(const Stream& stream,
                T *input,
                int num_tokens)
    {
        dim3 grid(num_tokens);
        dim3 block(std::min(dim, 1024));

        quant_kernel<T, half *, true><<<grid, block, 0, stream.stream>>>(input, output, output_scale, num_tokens, dim);

    }
};

template <typename T, bool use_per_token_quant>
struct QuantizerFuseSum
{
    int dim;
    int8_t* output;
    half* output_scale;
    half* output_sum;

    QuantizerFuseSum(int dim)
    {
        this->dim = dim;
    }

    int64_t init_output_ptr(Memory * memory, int32_t num_tokens, int64_t offset){
        return memory->allocate((void**)&this->output, offset, num_tokens * dim * sizeof(int8_t));
    }

    int64_t init_output_scale_and_sum_ptr(Memory * memory, int32_t num_tokens, int64_t offset){
        int64_t scale_offset = memory->allocate((void**)&this->output_scale, offset, num_tokens * sizeof(half));
        return memory->allocate((void**)&this->output_sum, scale_offset, num_tokens * sizeof(half));
    }

    void invoke(const Stream& stream,
                T *input,
                int num_tokens)
    {
        
        dim3 grid(num_tokens);
        dim3 block(std::min(dim, 1024));

        quant_kernel_fuse_sum<T, half *, true><<<grid, block, 0, stream.stream>>>(
        input, output, output_sum, output_scale, num_tokens, dim);

    }
};