#pragma once
#include "../w8a8/normq.cuh"
#include "../w8a8/fused_kernel.cuh"
#include "w4a8_qoq_group_linear.cuh"
#include "../linear.cuh"
#include <cuda_runtime.h>

template <typename T>
struct W4A8QoQGroupGatedFFN{
    int hidden_size;
    int intermediate_size;
    float rms_norm_eps;

    RMSNormQuant<T, true> *ffn_norm;
    W4A8QoQGroupLinear<T> *gate_up_proj;
    Quantizer<T, true> *down_quant_invoker;
    W4A8QoQGroupLinear<T> *down_proj;

    T* gated_up;

    W4A8QoQGroupGatedFFN(
        int hidden_size,
        int intermediate_size, 
        float rms_norm_eps
    ) {
        this->hidden_size = hidden_size;
        this->intermediate_size = intermediate_size;
        this->rms_norm_eps = rms_norm_eps;

        this->ffn_norm = new RMSNormQuant<T, true>(hidden_size, rms_norm_eps);
        this->gate_up_proj = new W4A8QoQGroupLinear<T>(hidden_size, intermediate_size*2);
        this->down_quant_invoker = new Quantizer<T, true>(intermediate_size);
        this->down_proj = new W4A8QoQGroupLinear<T>(intermediate_size, hidden_size);

    }

    void init_weight_ptr(Memory* memory) {
        this->ffn_norm->init_weight_ptr(memory);
        this->gate_up_proj->init_weight_ptr(memory);
        this->down_proj->init_weight_ptr(memory);

        this->gate_up_proj->init_s1_scales_ptr(memory);
        this->down_proj->init_s1_scales_ptr(memory);

        this->gate_up_proj->init_s2_scales_ptr(memory);
        this->down_proj->init_s2_scales_ptr(memory);

        this->gate_up_proj->init_s2_zeros_ptr(memory);
        this->down_proj->init_s2_zeros_ptr(memory);
    }
    
    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t ffn_norm_output_offset = this->ffn_norm->init_output_ptr(memory, num_tokens, offset);
        int64_t ffn_norm_output_scale_offset = this->ffn_norm->init_output_scale_ptr(memory, num_tokens, ffn_norm_output_offset);

        int64_t up_proj_end = this->gate_up_proj->init_output_ptr(memory, num_tokens, ffn_norm_output_scale_offset);
        int64_t gated_up_end = memory->allocate((void**)&this->gated_up, up_proj_end, num_tokens * intermediate_size * sizeof(T));

        int64_t invoke_output_offset = this->down_quant_invoker->init_output_ptr(memory, num_tokens, gated_up_end);
        int64_t invoke_output_scale_offset = this->down_quant_invoker->init_output_scale_ptr(memory, num_tokens, invoke_output_offset);

        int64_t down_proj_end = this->down_proj->init_output_ptr(memory, num_tokens, invoke_output_scale_offset);
        return down_proj_end;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("gate_up_proj") != std::string::npos) {
            this->gate_up_proj->load_to_storage(name, ptr);
        } else if (name.find("down_proj") != std::string::npos) {
            this->down_proj->load_to_storage(name, ptr);
        } else if (name.find("post_attention_layernorm") != std::string::npos) {
            this->ffn_norm->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, void* output=nullptr) {
        this->ffn_norm->prefill(stream, num_tokens, input);

        this->gate_up_proj->prefill(stream, num_tokens, this->ffn_norm->output, this->ffn_norm->output_scale);

        gated_silu_interleaved<T>(stream, num_tokens, this->intermediate_size, this->gate_up_proj->output, this->gated_up);

        this->down_quant_invoker->invoke(stream, this->gated_up, num_tokens);
        this->down_proj->prefill(stream, num_tokens, this->down_quant_invoker->output, this->down_quant_invoker->output_scale);
        elementwise_add<T>(stream, num_tokens, this->hidden_size, input, this->down_proj->output, input);
    }

    
};