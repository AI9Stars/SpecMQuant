#pragma once
#include "../norm.cuh"
#include "../activation.cuh"
#include "normq_sfloat.cuh"
#include "quant_sfloat.cuh"
#include "w4a8_qqq_linear.cuh"
#include <cuda_runtime.h>


template <typename T>
struct W4A8QQQFFN {
    int hidden_size;
    int intermediate_size;
    float rms_norm_eps;

    RMSNormQuantSfloat<T> *ffn_norm_quant;
    W4A8QQQLinear<T> *gate_up_proj;
    QuantizerScalefloat<T> * down_quantizer;
    W4A8QQQLinear<T> *down_proj;

    T* output;
    T* gated_up;

    W4A8QQQFFN(int hidden_size, int intermediate_size, float rms_norm_eps, int group_size) {
        this->hidden_size = hidden_size;
        this->intermediate_size = intermediate_size;
        this->rms_norm_eps = rms_norm_eps;

        this->ffn_norm_quant = new RMSNormQuantSfloat<T>(hidden_size, rms_norm_eps);
        this->gate_up_proj = new W4A8QQQLinear<T>(hidden_size, intermediate_size*2, group_size);
        this->down_quantizer = new QuantizerScalefloat<T>(intermediate_size);
        this->down_proj = new W4A8QQQLinear<T>(intermediate_size, hidden_size, group_size);
    }

    void init_weight_ptr(Memory* memory) {
        this->ffn_norm_quant->init_weight_ptr(memory);
        this->gate_up_proj->init_weight_ptr(memory);
        this->down_proj->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t ffn_norm_quant_end = this->ffn_norm_quant->init_output_ptr(memory, num_tokens, offset);
        int64_t up_proj_end = this->gate_up_proj->init_output_ptr(memory, num_tokens, ffn_norm_quant_end);

        int64_t gated_up_end = memory->allocate((void**)&this->gated_up, up_proj_end, num_tokens * intermediate_size * sizeof(T));

        int64_t down_proj_quant_end = this->down_quantizer->init_output_ptr(memory, num_tokens, gated_up_end);
        int64_t down_proj_end = this->down_proj->init_output_ptr(memory, num_tokens, down_proj_quant_end);
        this->output = this->down_proj->output;

        return down_proj_end;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("gate_up_proj") != std::string::npos) {
            this->gate_up_proj->load_to_storage(name, ptr);
        } else if (name.find("down_proj") != std::string::npos) {
            this->down_proj->load_to_storage(name, ptr);
        } else if (name.find("post_attention_layernorm") != std::string::npos) {
            this->ffn_norm_quant->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output) {

        this->ffn_norm_quant->prefill(stream, num_tokens, input, prev_output);
        this->gate_up_proj->prefill(stream, num_tokens, this->ffn_norm_quant->output, this->ffn_norm_quant->output_scale);

        gated_silu_interleaved<T>(stream, num_tokens, this->intermediate_size, this->gate_up_proj->output, this->gated_up);

        this->down_quantizer->invoke(stream, this->gated_up, num_tokens);
        this->down_proj->prefill(stream, num_tokens, this->down_quantizer->output, this->down_quantizer->output_scale);

    }

    void decode(const Stream& stream, int32_t num_tokens, T* input, T* prev_output) {
        prefill(stream, num_tokens, input, prev_output);
    }
};
