#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
// #include "../test_trait.cuh"
#include "../../utils.cuh"
// #include "../linear.cuh"
#include "../../qgemm/w4a8_qqq/w4a8_gemm_qqq.cuh"

template <typename T, bool transposed=true, bool has_bias=false>
struct W4A8QQQLinear{
    int32_t dim_in;
    int32_t dim_out;
    T* output;
    int32_t* B;
    float* s_channel;
    half* s_group;
    int32_t* workspace;
    T* bias;
    int group_size;
    int actual_size_n;
    
     // corresponding to gemm kernel
    const int max_par = 16;
    const int min_n_threads = 64;
    const int max_parallel = 16;
    const int pack_factor_4bit = 8;

    // tmp
    int32_t* c_tmp;

    W4A8QQQLinear(int dim_in, int dim_out, int group_size) {
        this->dim_in = dim_in;
        this->dim_out = dim_out;

        this->group_size = group_size;
        if (this->group_size == -1) {
            this->s_group = nullptr;
        }
    }

    void init_scale_ptr(Memory* memory) {
        s_channel = (float*)memory->allocate_for_model(dim_out * sizeof(float));
    }

    void init_workspace_ptr(Memory* memory) {
        int workspace_size = this->dim_out / this->min_n_threads * this->max_parallel;
        workspace = (int32_t*)memory->allocate_for_model(workspace_size * sizeof(int32_t));
    }

    void init_weight_ptr(Memory* memory) {
        const int w_size = this->dim_in * this->dim_out /pack_factor_4bit;
        B = (int32_t*)memory->allocate_for_model(w_size*sizeof(int32_t));
        if constexpr (has_bias) {
            bias = (T*)memory->allocate_for_model(dim_out * sizeof(T));
        }
        if (this->group_size > 0) {
            int s_group_size = this->dim_in / this->group_size * this->dim_out;
            s_group = (half*)memory->allocate_for_model(s_group_size * sizeof(half));
        }

        this->init_scale_ptr(memory);
        this->init_workspace_ptr(memory);
        
    }
    

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t output_offset =  memory->allocate((void**)&this->output, offset, num_tokens * dim_out * sizeof(T));
        // int64_t tmp_offset = this->init_tmp_ptr(memory, num_tokens, output_offset);
        int64_t c_tmp_offset = memory->allocate((void**)&this->c_tmp, output_offset, this->max_par * this->dim_out * 64 * sizeof(int32_t));
        return c_tmp_offset;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("s_channel") != std::string::npos) {
            cudaMemcpy((void*)s_channel, ptr, this->dim_out*sizeof(float), cudaMemcpyHostToDevice);
        } else if (name.find(".B") != std::string::npos) {
            const int w_size = this->dim_in * this->dim_out / this->pack_factor_4bit;
            cudaMemcpy((void*)B, ptr, w_size*sizeof(int32_t), cudaMemcpyHostToDevice);
        } else if (name.find("s_group") != std::string::npos) {
            if (this->group_size == -1) {
                return;
            }
            int s_group_size = this->dim_in / this->group_size * this->dim_out;
            cudaMemcpy((void*)s_group, ptr, s_group_size*sizeof(half), cudaMemcpyHostToDevice);
        } else if (name.find("bias") != std::string::npos) {
            cudaMemcpy((void*)bias, ptr, dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }
    
    void prefill(const Stream& stream, int32_t num_tokens, int8_t* input, float* input_scales, T* tgt=nullptr) {
        if (tgt == nullptr) tgt = this->output;
        
        marlin_qqq_gemm(
            input,
            this->B,
            input_scales,
            this->s_channel,
            this->s_group,
            this->workspace, num_tokens,
            this->dim_out, this->dim_in,
            this->group_size,
            this->c_tmp,
            tgt,
            stream.stream
        );


        // if constexpr (has_bias) {
        //     batched_add<T>(stream, num_tokens, this->dim_out, tgt, this->bias, tgt);
        // }
    }



};