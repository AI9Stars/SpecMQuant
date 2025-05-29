#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "../../qgemm/w8a8/w8a8_gemm_cuda.cuh"

template <typename T, bool transposed=true, bool has_bias=false>
struct W8A8Linear{
    int dim_in;
    int dim_out;
    T* output;
    int8_t* weight;
    T* bias;
    half2* weight_scale;

    W8A8Linear(int dim_in, int dim_out) {
        this->dim_in = dim_in;
        this->dim_out = dim_out;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (int8_t*)memory->allocate_for_model(dim_in * dim_out * sizeof(int8_t));
        if constexpr (has_bias) {
            bias = (T*)memory->allocate_for_model(dim_out * sizeof(T));
        }
    }

    void init_weight_scale_ptr(Memory* memory) {
        weight_scale = (half2*)memory->allocate_for_model(dim_out * sizeof(half));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * dim_out * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("dequant_scale") != std::string::npos) {
            cudaMemcpy((void*)weight_scale, ptr, dim_out * sizeof(half), cudaMemcpyHostToDevice);
        } else if (name.find("weight") != std::string::npos) {
            cudaMemcpy((void*)weight, ptr, dim_in * dim_out * sizeof(int8_t), cudaMemcpyHostToDevice);
        } else if (name.find("bias") != std::string::npos) {
            cudaMemcpy((void*)bias, ptr, dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, int8_t* input, half* input_scale, T* tgt=nullptr) {
        if (tgt == nullptr) tgt = this->output;
        w8a8_gemm_forward_cuda(
            stream,
            input,
            weight,
            weight_scale,
            input_scale,
            (half*)tgt,
            num_tokens,
            dim_in,
            num_tokens,
            dim_out
        );
    }
        
};