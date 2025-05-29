#pragma once
#include <cuda_runtime.h>
#include "../../trait.cuh"
#include "../../utils.cuh"
#include "../../utilsq.cuh"

namespace {
template <typename T, typename T2>
__global__ void rms_norm_quant_sfloat_kernel(int dim, const T2* input, const T2* weight, int8_t* output, float*scale,  float eps) {
    // __shared__ T2 s_input[2048];
    __shared__ float shared_sum;
    __shared__ float warp_sum[16];

    // for quant
    __shared__ float shared_absmax_val;
    __shared__ float warp_absmax_val[16];
    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 val = input[row * dim + i];
        // s_input[i] = val;
        float val1 = float(val.x);
        float val2 = float(val.y);
        sum1 += val1 * val1;
        sum2 += val2 * val2;
    }
    float sum = sum1 + sum2;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (col % 32 == 0) warp_sum[col / 32] = sum;
    __syncthreads();
    if (col < 16) {
        sum = warp_sum[col];
        sum += __shfl_down_sync(0x0000ffff, sum, 8);
        sum += __shfl_down_sync(0x0000ffff, sum, 4);
        sum += __shfl_down_sync(0x0000ffff, sum, 2);
        sum += __shfl_down_sync(0x0000ffff, sum, 1);
    }
    if (col == 0) {
        shared_sum = rsqrtf(sum / (2*dim) + eps);
    }
    __syncthreads();
    sum = shared_sum;
    // quant
    float absmax_val1 = 0.0f;
    float absmax_val2 = 0.0f;
    float const zero = 0.0f;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 inp = input[row * dim + i];
        T2 w = weight[i];
        float val1 = sum * float(inp.x) * float(w.x);
        float val2 = sum * float(inp.y) * float(w.y);
        val1 = val1 > zero ? val1 : -val1;
        val2 = val2 > zero ? val2 : -val2;
        absmax_val1 = val1 > absmax_val1 ? val1 : absmax_val1;
        absmax_val2 = val2 > absmax_val2 ? val2 : absmax_val2;
    }
    float absmax_val = absmax_val1 > absmax_val2 ? absmax_val1 : absmax_val2;
    // TODO: remove the warp size=32
    absmax_val = max(absmax_val, __shfl_xor_sync(0xffffffff, absmax_val, 16, 32));
    absmax_val = max(absmax_val, __shfl_xor_sync(0xffffffff, absmax_val, 8, 32));
    absmax_val = max(absmax_val, __shfl_xor_sync(0xffffffff, absmax_val, 4, 32));
    absmax_val = max(absmax_val, __shfl_xor_sync(0xffffffff, absmax_val, 2, 32));
    absmax_val = max(absmax_val, __shfl_xor_sync(0xffffffff, absmax_val, 1, 32));
    if (col % 32 == 0) warp_absmax_val[col / 32] = absmax_val;
    __syncthreads();
    // v1 implement
    if (col < 16) {
        absmax_val = warp_absmax_val[col];
        absmax_val = max(absmax_val, __shfl_xor_sync(0x0000ffff, absmax_val, 8, 32));
        absmax_val = max(absmax_val, __shfl_xor_sync(0x0000ffff, absmax_val, 4, 32));
        absmax_val = max(absmax_val, __shfl_xor_sync(0x0000ffff, absmax_val, 2, 32));
        absmax_val = max(absmax_val, __shfl_xor_sync(0x0000ffff, absmax_val, 1, 32));
    }
    if (col == 0) {
        shared_absmax_val = absmax_val;
        scale[row] = absmax_val / 127.0f;
    }
    __syncthreads();
    absmax_val = shared_absmax_val;
    // v2 implement 
    // absmax_val = warp_shared_val[col%16]; 
    // absmax_val = max(absmax_val, __shfl_down_sync(0xffffffff, absmax_val, 8, 32));
    // absmax_val = max(absmax_val, __shfl_down_sync(0xffffffff, absmax_val, 4, 32));
    // absmax_val = max(absmax_val, __shfl_down_sync(0xffffffff, absmax_val, 2, 32));
    // absmax_val = max(absmax_val, __shfl_down_sync(0xffffffff, absmax_val, 1, 32));
    float const tmp_scale = 127.0f /absmax_val;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 inp_q = input[row * dim + i];
        T2 w_q = weight[i];
        float val1_q = sum * float(inp_q.x) * float(w_q.x) * tmp_scale;
        float val2_q = sum * float(inp_q.y) * float(w_q.y) * tmp_scale;
        output[(row * dim + i) * 2] = float_to_int8_rn(val1_q);
        output[(row * dim + i) * 2 + 1] = float_to_int8_rn(val2_q);
    }
    
}

template <typename T, typename T2>
__global__ void add_and_rms_norm_quant_sfloat_kernel(int dim, T2* input, const T2* prev_output, const T2* weight, int8_t* output, float* scale, float eps) {
    // __shared__ T2 s_input[2048];
    __shared__ float shared_sum;
    __shared__ float warp_sum[16];

    // for quant
    __shared__ float shared_absmax_val;
    __shared__ float warp_absmax_val[16];
    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 val = input[row * dim + i];
        T2 prev = prev_output[row * dim + i];
        val = val + prev;
        input[row * dim + i] = val;
        float val1 = float(val.x);
        float val2 = float(val.y);
        sum1 += val1 * val1;
        sum2 += val2 * val2;
    }
    float sum = sum1 + sum2;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (col % 32 == 0) warp_sum[col / 32] = sum;
    __syncthreads();
    if (col < 16) {
        sum = warp_sum[col];
        sum += __shfl_down_sync(0x0000ffff, sum, 8);
        sum += __shfl_down_sync(0x0000ffff, sum, 4);
        sum += __shfl_down_sync(0x0000ffff, sum, 2);
        sum += __shfl_down_sync(0x0000ffff, sum, 1);
    }
    if (col == 0) {
        shared_sum = rsqrtf(sum / (2*dim) + eps);
    }
    __syncthreads();
    sum = shared_sum;
    float absmax_val1 = 0.0f;
    float absmax_val2 = 0.0f;
    float const zero = 0.0f;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 inp = input[row * dim + i];
        T2 w = weight[i];
        float val1 = sum * float(inp.x) * float(w.x);
        float val2 = sum * float(inp.y) * float(w.y);
        val1 = val1 > zero ? val1 : -val1;
        val2 = val2 > zero ? val2 : -val2;
        absmax_val1 = val1 > absmax_val1 ? val1 : absmax_val1;
        absmax_val2 = val2 > absmax_val2 ? val2 : absmax_val2;
    }
    float absmax_val = absmax_val1 > absmax_val2 ? absmax_val1 : absmax_val2;
    // TODO: remove the warp size=32
    absmax_val = max(absmax_val, __shfl_xor_sync(0xffffffff, absmax_val, 16, 32));
    absmax_val = max(absmax_val, __shfl_xor_sync(0xffffffff, absmax_val, 8, 32));
    absmax_val = max(absmax_val, __shfl_xor_sync(0xffffffff, absmax_val, 4, 32));
    absmax_val = max(absmax_val, __shfl_xor_sync(0xffffffff, absmax_val, 2, 32));
    absmax_val = max(absmax_val, __shfl_xor_sync(0xffffffff, absmax_val, 1, 32));
    if (col % 32 == 0) warp_absmax_val[col / 32] = absmax_val;
    __syncthreads();
    // v1 implement
    if (col < 16) {
        absmax_val = warp_absmax_val[col];
        absmax_val = max(absmax_val, __shfl_xor_sync(0x0000ffff, absmax_val, 8, 32));
        absmax_val = max(absmax_val, __shfl_xor_sync(0x0000ffff, absmax_val, 4, 32));
        absmax_val = max(absmax_val, __shfl_xor_sync(0x0000ffff, absmax_val, 2, 32));
        absmax_val = max(absmax_val, __shfl_xor_sync(0x0000ffff, absmax_val, 1, 32));
    }
    if (col == 0) {
        shared_absmax_val = absmax_val;
        scale[row] = absmax_val / 127.0f;
    }
    __syncthreads();
    absmax_val = shared_absmax_val;
    // v2 implement 
    // absmax_val = warp_shared_val[col%16]; 
    // absmax_val = max(absmax_val, __shfl_down_sync(0xffffffff, absmax_val, 8, 32));
    // absmax_val = max(absmax_val, __shfl_down_sync(0xffffffff, absmax_val, 4, 32));
    // absmax_val = max(absmax_val, __shfl_down_sync(0xffffffff, absmax_val, 2, 32));
    // absmax_val = max(absmax_val, __shfl_down_sync(0xffffffff, absmax_val, 1, 32));
    float const tmp_scale = 127.0f /absmax_val;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 inp_q = input[row * dim + i];
        T2 w_q = weight[i];
        float val1_q = sum * float(inp_q.x) * float(w_q.x) * tmp_scale;
        float val2_q = sum * float(inp_q.y) * float(w_q.y) * tmp_scale;
        output[(row * dim + i) * 2] = float_to_int8_rn(val1_q);
        output[(row * dim + i) * 2 + 1] = float_to_int8_rn(val2_q);
    }
}

template <typename T>
void rms_norm_quant_sfloat(const Stream& stream, int num_tokens, int dim, const T* input, const T* weight, int8_t* output, float* scale, float eps) {
    using T2 = typename TypeTraits<T>::half2;
    rms_norm_quant_sfloat_kernel<T, T2><<<num_tokens, 512, 0, stream.stream>>>(dim/2, (T2*)input, (T2*)weight, output, scale, eps);
}

template <typename T>
void add_and_rms_norm_quant_sfloat(const Stream& stream, int num_tokens, int dim, T* input, const T* prev_output, const T* weight, int8_t* output, float* scale, float eps) {
    using T2 = typename TypeTraits<T>::half2;
    add_and_rms_norm_quant_sfloat_kernel<T, T2><<<num_tokens, 512, 0, stream.stream>>>(dim/2, (T2*)input, (T2*)prev_output, (T2*)weight, output, scale, eps);
}
}


template <typename T>
struct RMSNormQuantSfloat {
    int dim;
    float eps;
    T* weight;

    int8_t* output;
    float* output_scale;

    RMSNormQuantSfloat(int dim, float eps) {
        this->dim = dim;
        this->eps = eps;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(dim * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t output_offset = memory->allocate((void**)&this->output, offset, num_tokens * dim * sizeof(int8_t));
        int64_t scale_offset = memory->allocate((void**)&this->output_scale, output_offset, num_tokens * sizeof(float));
        return scale_offset;
    }

    void load_to_storage(std::string name, void* ptr) {
        cudaMemcpy((void*)weight, ptr, dim * sizeof(T), cudaMemcpyHostToDevice);
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output, int8_t* tgt=nullptr) {
        if (tgt == nullptr) tgt = this->output;
        if (prev_output == nullptr) {
            rms_norm_quant_sfloat(stream, num_tokens, this->dim, input, this->weight, tgt, this->output_scale, this->eps);
        } else {
            add_and_rms_norm_quant_sfloat(stream, num_tokens, this->dim, input, prev_output, this->weight, tgt, this->output_scale, this->eps);
        }
    }
};