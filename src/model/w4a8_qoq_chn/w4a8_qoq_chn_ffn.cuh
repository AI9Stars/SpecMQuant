#pragma once
#include "../w8a8/normq.cuh"
#include "../w8a8/fused_kernel.cuh"
#include "w4a8_qoq_chn_linear.cuh"
#include "../linear.cuh"
#include <cuda_runtime.h>

template <typename T>
struct W4A8QoQChnGatedFFN{
    int hidden_size;
    int intermediate_size;
    float rms_norm_eps;

    RMSNormQuantFuseSum<T, true> *ffn_norm;
    W4A8QoQChnLinear<T> *gate_proj, *up_proj;
    QuantizerFuseSum<T, true> *down_quant_invoker;
    W4A8QoQChnLinear<T> *down_proj;

    T* gated_up;

    W4A8QoQChnGatedFFN(
        int hidden_size,
        int intermediate_size, 
        float rms_norm_eps
    ) {
        this->hidden_size = hidden_size;
        this->intermediate_size = intermediate_size;
        this->rms_norm_eps = rms_norm_eps;

        this->ffn_norm = new RMSNormQuantFuseSum<T, true>(hidden_size, rms_norm_eps);
        this->gate_proj = new W4A8QoQChnLinear<T>(hidden_size, intermediate_size);
        this->up_proj = new W4A8QoQChnLinear<T>(hidden_size, intermediate_size);
        this->down_quant_invoker = new QuantizerFuseSum<T, true>(intermediate_size);
        this->down_proj = new W4A8QoQChnLinear<T>(intermediate_size, hidden_size);

    }

    void init_weight_ptr(Memory* memory) {
        this->ffn_norm->init_weight_ptr(memory);
        this->gate_proj->init_weight_ptr(memory);
        this->up_proj->init_weight_ptr(memory);
        this->down_proj->init_weight_ptr(memory);

        // init weight scale for linear
        this->gate_proj->init_s1_scales_ptr(memory);
        this->up_proj->init_s1_scales_ptr(memory);
        this->down_proj->init_s1_scales_ptr(memory);

        // init zeros for quantizer
        this->gate_proj->init_s1_szeros_ptr(memory);
        this->up_proj->init_s1_szeros_ptr(memory);
        this->down_proj->init_s1_szeros_ptr(memory);
    }
    
    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t ffn_norm_output_offset = this->ffn_norm->init_output_ptr(memory, num_tokens, offset);
        int64_t ffn_norm_output_scale_offset = this->ffn_norm->init_output_scale_and_sum_ptr(memory, num_tokens, ffn_norm_output_offset);

        int64_t gate_proj_end = this->gate_proj->init_output_ptr(memory, num_tokens, ffn_norm_output_scale_offset);
        int64_t up_proj_end = this->up_proj->init_output_ptr(memory, num_tokens, gate_proj_end);
        int64_t gated_up_end = memory->allocate((void**)&this->gated_up, up_proj_end, num_tokens * intermediate_size * sizeof(T));

        int64_t invoke_output_offset = this->down_quant_invoker->init_output_ptr(memory, num_tokens, gated_up_end);
        int64_t invoke_output_scale_offset = this->down_quant_invoker->init_output_scale_and_sum_ptr(memory, num_tokens, invoke_output_offset);

        int64_t down_proj_end = this->down_proj->init_output_ptr(memory, num_tokens, invoke_output_scale_offset);
        return down_proj_end;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("gate_proj") != std::string::npos) {
            this->gate_proj->load_to_storage(name, ptr);
        } else if (name.find("up_proj") != std::string::npos) {
            this->up_proj->load_to_storage(name, ptr);
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
        // if (output == nullptr) {
        //     output = this->gated_up;
        // }
        w4a8_qoq_chn_gemm_forward_cuda(
            stream,
            this->ffn_norm->output,
            this->gate_proj->weight,
            this->gate_proj->s1_scales,
            this->ffn_norm->output_scale,
            this->gate_proj->s1_szeros,
            this->ffn_norm->output_sum,
            (half*)this->gate_proj->output,
            num_tokens,
            hidden_size,
            num_tokens,
            intermediate_size*2
        );
        gated_silu_interleaved<T>(stream, num_tokens, this->intermediate_size, this->gate_proj->output, this->gated_up);

        this->down_quant_invoker->invoke(stream, this->gated_up, num_tokens);
        this->down_proj->prefill(stream, num_tokens, this->down_quant_invoker->output, this->down_quant_invoker->output_scale, this->down_quant_invoker->output_sum);
        elementwise_add<T>(stream, num_tokens, this->hidden_size, input, this->down_proj->output, input);
    }

    
};