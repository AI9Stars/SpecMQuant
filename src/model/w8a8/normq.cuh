#pragma once
#include "../../reduction_utils.cuh"
#include <cuda_fp16.h>
#include <cassert>
#include "../../utilsq.cuh"
#include "../../utils.cuh"


// from qserve & vllm
namespace {
// from TRTLLM
template <typename Tf, typename T>
__inline__ __device__ Tf compute_layernorm(Tf val, float s_mean, float s_variance, const T* gamma, const T* beta, int i)
{
    Tf ret = (val - s_mean) * s_variance * cuda_cast<Tf>(gamma[i]);
    if (beta != nullptr)
    {
        ret = ret + cuda_cast<Tf>(beta[i]);
    }
    return ret;
}


template <typename T, typename scale_type, bool USE_DIFF_OF_SQUARES = false>
__global__ void generalLayerNorm(T* input, T* gamma, T* beta, T* normed_output, float eps,
    int tokens, int hidden_dim, scale_type* scale_orig_quant_per_tensor, scale_type* scale_orig_quant_per_token,
    int8_t* normed_output_quant, bool use_shmem)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float s_mean;
    __shared__ float s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float mean = 0.0f;
    float variance = 0.0f;
    float local_sum = 0.0f;
    float local_var_sum = 0.0f;

    const int n_elems = hidden_dim / num_elems_T;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const T val = input[bidx * n_elems + i];
        if (use_shmem)
        {
            shmem[i] = val;
        }

        const float_packed_t val_f = cuda_cast<float_packed_t>(val);
        local_sum += cuda_sum<float>(val_f);
        if (USE_DIFF_OF_SQUARES)
        {
            local_var_sum += cuda_sum<float>(val_f * val_f);
        }
    }

    if (USE_DIFF_OF_SQUARES)
    {
        float packed[2] = {local_sum, local_var_sum};
        blockReduceSumV2<float, 2>(packed);
        mean = packed[0];
        variance = packed[1];
    }
    else
    {
        mean = blockReduceSum(local_sum);
    }

    if (threadIdx.x == 0)
    {
        mean = mean / hidden_dim;
        s_mean = mean;
        if (USE_DIFF_OF_SQUARES)
        {
            variance = (variance / hidden_dim) - (mean * mean); // Var[x] = E[x²] - E[x]²
            s_variance = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    if (!USE_DIFF_OF_SQUARES)
    {
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const T val = use_shmem ? shmem[i] : input[bidx * n_elems + i];
            float_packed_t diff = cuda_cast<float_packed_t>(val) - s_mean;
            local_var_sum += cuda_sum<float>(diff * diff);
        }
        variance = blockReduceSum(local_var_sum);

        if (threadIdx.x == 0)
        {
            s_variance = rsqrtf(variance / hidden_dim + eps);
        }
        __syncthreads();
    }

    const bool with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    const bool with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    const float_packed_t scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? __half2float(*scale_orig_quant_per_tensor) : 0.0f);
    T_scalar amax = 1e-6f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
        const T val = cuda_cast<T>(compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i));

        if (with_per_token_scaling)
        {
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            if (use_shmem)
            {
                shmem[i] = val;
            }
        }
        else if (with_per_tensor_scaling)
        {
            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        const float dynamic_per_token_scale = 127.f / abs_max_f;
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const int index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
            if (!use_shmem)
            {
                val_f = compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i);
            }

            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = abs_max_f / 127.f;
        }
    }
}

template <typename T, typename scale_type>
__global__ void generalRMSNorm(const T* input, const T* gamma, const T* beta, T* normed_output, const float eps,
    int tokens, int hidden_dim, const scale_type* scale_orig_quant_per_tensor, scale_type* scale_orig_quant_per_token,
    int8_t* normed_output_quant, bool use_shmem)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float s_mean;
    __shared__ float s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float variance = 0.0f;
    float local_var_sum = 0.0f;

    const int n_elems = hidden_dim / num_elems_T;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const T val = use_shmem ? shmem[i] : input[bidx * n_elems + i];
        float_packed_t diff = cuda_cast<float_packed_t>(val); // no mean
        local_var_sum += cuda_sum<float>(diff * diff);
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0)
    {
        s_variance = rsqrtf(variance / hidden_dim + eps);
    }
    __syncthreads();

    const bool with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    const bool with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    const float_packed_t scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? __half2float(*scale_orig_quant_per_tensor) : 0.0f);
    T_scalar amax = 1e-6f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
        const T val = cuda_cast<T>(compute_layernorm(val_f, 0.0f, s_variance, gamma, beta, i));

        if (with_per_token_scaling)
        {
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            if (use_shmem)
            {
                shmem[i] = val;
            }
        }
        else if (with_per_tensor_scaling)
        {
            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        const float dynamic_per_token_scale = 127.f / abs_max_f;
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const int index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
            if (!use_shmem)
            {
                val_f = compute_layernorm(val_f, 0.0f, s_variance, gamma, beta, i);
            }

            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = abs_max_f / 127.f;
        }
    }
}

template <typename T, typename scale_type, bool USE_DIFF_OF_SQUARES = false>
__global__ void generalLayerNorm_fuse_sum(T* input, T* gamma, T* beta, T* normed_output, float eps,
    int tokens, int hidden_dim, scale_type* input_sum, scale_type* scale_orig_quant_per_tensor, scale_type* scale_orig_quant_per_token,
    int8_t* normed_output_quant, bool use_shmem)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float s_mean;
    __shared__ float s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float mean = 0.0f;
    float variance = 0.0f;
    float local_sum = 0.0f;
    float local_var_sum = 0.0f;

    const int n_elems = hidden_dim / num_elems_T;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const T val = input[bidx * n_elems + i];
        if (use_shmem)
        {
            shmem[i] = val;
        }

        const float_packed_t val_f = cuda_cast<float_packed_t>(val);
        local_sum += cuda_sum<float>(val_f);
        if (USE_DIFF_OF_SQUARES)
        {
            local_var_sum += cuda_sum<float>(val_f * val_f);
        }
    }

    if (USE_DIFF_OF_SQUARES)
    {
        float packed[2] = {local_sum, local_var_sum};
        blockReduceSumV2<float, 2>(packed);
        mean = packed[0];
        variance = packed[1];
    }
    else
    {
        mean = blockReduceSum(local_sum);
    }

    if (threadIdx.x == 0)
    {
        mean = mean / hidden_dim;
        s_mean = mean;
        if (USE_DIFF_OF_SQUARES)
        {
            variance = (variance / hidden_dim) - (mean * mean); // Var[x] = E[x²] - E[x]²
            s_variance = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    if (!USE_DIFF_OF_SQUARES)
    {
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const T val = use_shmem ? shmem[i] : input[bidx * n_elems + i];
            float_packed_t diff = cuda_cast<float_packed_t>(val) - s_mean;
            local_var_sum += cuda_sum<float>(diff * diff);
        }
        variance = blockReduceSum(local_var_sum);

        if (threadIdx.x == 0)
        {
            s_variance = rsqrtf(variance / hidden_dim + eps);
        }
        __syncthreads();
    }

    const bool with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    const bool with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    const float_packed_t scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? __half2float(*scale_orig_quant_per_tensor) : 0.0f);
    T_scalar amax = 1e-6f;
    T_scalar sum = 0.0f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
        const T val = cuda_cast<T>(compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i));

        if (with_per_token_scaling)
        {
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            sum += cuda_sum<float>(val);
            if (use_shmem)
            {
                shmem[i] = val;
            }
        }
        else if (with_per_tensor_scaling)
        {
            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        float sum_f = blockAllReduceSum(cuda_cast<float>(sum));
        const float dynamic_per_token_scale = 127.f / abs_max_f;
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const int index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
            if (!use_shmem)
            {
                val_f = compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i);
            }

            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = abs_max_f / 127.f;
            input_sum[bidx] = sum_f;
        }
    }
}

template <typename T, typename scale_type, bool USE_DIFF_OF_SQUARES = false>
__global__ void generalRMSNorm_fuse_sum(const T* input, const T* gamma, const T* beta, T* normed_output, const float eps,
    int tokens, int hidden_dim, scale_type* input_sum, const scale_type* scale_orig_quant_per_tensor, scale_type* scale_orig_quant_per_token,
    int8_t* normed_output_quant, bool use_shmem)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float s_mean;
    __shared__ float s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float variance = 0.0f;
    float local_var_sum = 0.0f;

    const int n_elems = hidden_dim / num_elems_T;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const T val = use_shmem ? shmem[i] : input[bidx * n_elems + i];
        float_packed_t diff = cuda_cast<float_packed_t>(val); // no mean
        local_var_sum += cuda_sum<float>(diff * diff);
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0)
    {
        s_variance = rsqrtf(variance / hidden_dim + eps);
    }
    __syncthreads();

    const bool with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    const bool with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    const float_packed_t scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? __half2float(*scale_orig_quant_per_tensor) : 0.0f);
    T_scalar amax = 1e-6f;
    T_scalar sum = 0.0f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
        const T val = cuda_cast<T>(compute_layernorm(val_f, 0.0f, s_variance, gamma, beta, i));

        if (with_per_token_scaling)
        {
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            sum += cuda_sum<float>(val);
            if (use_shmem)
            {
                shmem[i] = val;
            }
        }
        else if (with_per_tensor_scaling)
        {
            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        float sum_f = blockAllReduceSum(cuda_cast<float>(sum));
        const float dynamic_per_token_scale = 127.f / abs_max_f;
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const int index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
            if (!use_shmem)
            {
                val_f = compute_layernorm(val_f, 0.0f, s_variance, gamma, beta, i);
            }

            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = abs_max_f / 127.f;
            input_sum[bidx] = sum_f;
        }
    }
}

}



template <typename T, bool use_per_token_quant>
struct RMSNormQuant {
    int dim;
    float eps;
    T* weight;
    int8_t* output;
    __half* output_scale;

    RMSNormQuant(int dim, float eps) {
        this->dim = dim;
        this->eps = eps;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(dim * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * dim * sizeof(int8_t));
    }
    int64_t init_output_scale_ptr(Memory * memory, int32_t num_tokens, int64_t offset){
        return memory->allocate((void**)&this->output_scale, offset, num_tokens * sizeof(__half));
    }

    void load_to_storage(std::string name, void* ptr) {
        cudaMemcpy((void*)weight, ptr, dim * sizeof(T), cudaMemcpyHostToDevice);
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, half* tgt=nullptr) {
        dim3 grid(num_tokens);
        dim3 block(std::min(dim, 1024));
        block.x = 32 * ((block.x + 31) / 32);
        if (tgt==nullptr) tgt = this->output_scale;
        // using T = typename FloatTypeConverter<scalar_t>::Type;
        if (use_per_token_quant) {
            // per-token
            generalLayerNorm<T, half><<<grid, block, 0, stream.stream>>>(
            input, 
            weight, nullptr,
            nullptr, eps, num_tokens, dim, nullptr, output_scale,
            output, false
            );
            // input, gamma, beta, normed_output, eps, tokens, hidden_dim, per_tensor_scale, per_token_scale
            // normed_output_quant, use_shmem
            // out.data_ptr<int8_t>(), input.data_ptr<scalar_t>(),
            // weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
        } else {
            // per-tensor
            // not support
            generalLayerNorm<T, half><<<grid, block, 0, stream.stream>>>(
            input, 
            weight, nullptr,
            nullptr, eps, num_tokens, dim, output_scale, nullptr,
            output, false
            );
    }
    }


};

template <typename T, bool use_per_token_quant>
struct RMSNormQuantFuseSum {
    int dim;
    float eps;
    T* weight;
    int8_t* output;
    __half* output_scale;
    __half* output_sum;

    RMSNormQuantFuseSum(int dim, float eps) {
        this->dim = dim;
        this->eps = eps;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(dim * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * dim * sizeof(int8_t));
    }

    int64_t init_output_scale_and_sum_ptr(Memory * memory, int32_t num_tokens, int64_t offset){
        int64_t scale_end = memory->allocate((void**)&this->output_scale, offset, num_tokens * sizeof(__half));
        return memory->allocate((void**)&this->output_sum, scale_end, num_tokens * sizeof(__half));
    }

    void load_to_storage(std::string name, void* ptr) {
        cudaMemcpy((void*)weight, ptr, dim * sizeof(T), cudaMemcpyHostToDevice);
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, half* tgt=nullptr) {
        dim3 grid(num_tokens);
        dim3 block(std::min(dim, 1024));
        block.x = 32 * ((block.x + 31) / 32);
        
        if (use_per_token_quant) {
            // per-token
            generalLayerNorm_fuse_sum<T, half><<<grid, block, 0, stream.stream>>>(
            input, 
            weight, nullptr,
            nullptr, eps, num_tokens, dim, output_sum, nullptr, output_scale,
            output, false
            );
            
        } else {
            // per-tensor
            generalLayerNorm_fuse_sum<T, half><<<grid, block, 0, stream.stream>>>(
            input, 
            weight, nullptr,
            nullptr, eps, num_tokens, dim, nullptr, output_scale, nullptr,
            output, false
            );
        }
    }
};