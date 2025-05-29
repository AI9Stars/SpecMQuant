#pragma once
#include "../attn.cuh"
#include "../w8a8/fused_kernel.cuh"
#include "../w8a8/normq.cuh"
#include "w4a8_qoq_chn_linear.cuh"

template <typename T>
struct W4A8QoQChnAttention {
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    float rms_norm_eps;

    RMSNormQuantFuseSum<T, true> *attn_norm;
    W4A8QoQChnLinear<T> *q_proj, *k_proj, *v_proj;
    QuantizerFuseSum<T, true> *o_quant_invoker;
    W4A8QoQChnLinear<T> *o_proj;

    T* attn_output;
    float *softmax_lse, *softmax_lse_accum, *oaccum;

    W4A8QoQChnAttention(int hidden_size, int num_attention_heads, int num_key_value_heads, int head_dim, float rms_norm_eps) {
        this->hidden_size = hidden_size;
        this->num_attention_heads = num_attention_heads;
        this->num_key_value_heads = num_key_value_heads;
        this->head_dim = head_dim;
        this->rms_norm_eps = rms_norm_eps;

        this->attn_norm = new RMSNormQuantFuseSum<T, true>(hidden_size, rms_norm_eps);
        this->q_proj = new W4A8QoQChnLinear<T>(hidden_size, num_attention_heads * head_dim);
        this->k_proj = new W4A8QoQChnLinear<T>(hidden_size, num_key_value_heads * head_dim);
        this->v_proj = new W4A8QoQChnLinear<T>(hidden_size, num_key_value_heads * head_dim);
        this->o_quant_invoker = new QuantizerFuseSum<T, true>(hidden_size);
        this->o_proj = new W4A8QoQChnLinear<T>(hidden_size, num_attention_heads * head_dim);
    }

    void init_weight_ptr(Memory* memory) {
        this->attn_norm->init_weight_ptr(memory);
        this->q_proj->init_weight_ptr(memory);
        this->k_proj->init_weight_ptr(memory);
        this->v_proj->init_weight_ptr(memory);
        this->o_proj->init_weight_ptr(memory);

        // weight scale init
        this->q_proj->init_s1_scales_ptr(memory);
        this->k_proj->init_s1_scales_ptr(memory);
        this->v_proj->init_s1_scales_ptr(memory);
        this->o_proj->init_s1_scales_ptr(memory);

        // s1 zeros init
        this->q_proj->init_s1_szeros_ptr(memory);
        this->k_proj->init_s1_szeros_ptr(memory);
        this->v_proj->init_s1_szeros_ptr(memory);
        this->o_proj->init_s1_szeros_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t attn_norm_output_offset = this->attn_norm->init_output_ptr(memory, num_tokens, offset);
        int64_t attn_norm_output_scale_offset = this->attn_norm->init_output_scale_and_sum_ptr(memory, num_tokens, attn_norm_output_offset);

        int64_t q_proj_end = this->q_proj->init_output_ptr(memory, num_tokens, attn_norm_output_scale_offset);
        int64_t k_proj_end = this->k_proj->init_output_ptr(memory, num_tokens, q_proj_end);
        int64_t v_proj_end = this->v_proj->init_output_ptr(memory, num_tokens, k_proj_end);
        
        memory->allocate((void**)&this->attn_output, offset);
        int64_t softmax_lse_end = memory->allocate((void**)&this->softmax_lse, v_proj_end, num_tokens * this->num_attention_heads * sizeof(float));
        int64_t softmax_lse_accum_end = memory->allocate((void**)&this->softmax_lse_accum, softmax_lse_end, num_tokens * this->num_attention_heads * sizeof(float));
        int64_t oaccum_end = memory->allocate((void**)&this->oaccum, softmax_lse_accum_end, num_tokens * this->num_attention_heads * this->head_dim * sizeof(float));

        int64_t invoke_output_offset =  this->o_quant_invoker->init_output_ptr(memory, num_tokens, oaccum_end);
        int64_t invoke_output_scale_offset = this->o_quant_invoker->init_output_scale_and_sum_ptr(memory, num_tokens, invoke_output_offset);
        int64_t o_proj_end = this->o_proj->init_output_ptr(memory, num_tokens, invoke_output_scale_offset);

        return o_proj_end;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("q_proj") != std::string::npos) {
            this->q_proj->load_to_storage(name, ptr);
        } else if (name.find("k_proj") != std::string::npos) {
            this->k_proj->load_to_storage(name, ptr);
        } else if (name.find("v_proj") != std::string::npos) {
            this->v_proj->load_to_storage(name, ptr);
        } else if (name.find("o_proj") != std::string::npos) {
            this->o_proj->load_to_storage(name, ptr);
        } else if (name.find("input_layernorm") != std::string::npos) {
            this->attn_norm->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, int32_t num_history_tokens, T* input, int32_t* position_ids, KVCache<T>* kv_cache) {
        T* k_cache = kv_cache->offset_k(num_history_tokens);
        T* v_cache = kv_cache->offset_v(num_history_tokens);

        this->attn_norm->prefill(stream, num_tokens, input);
        this->q_proj->prefill(stream, num_tokens, this->attn_norm->output, this->attn_norm->output_scale, this->attn_norm->output_sum);
        this->k_proj->prefill(stream, num_tokens, this->attn_norm->output, this->attn_norm->output_scale, this->attn_norm->output_sum, k_cache);
        this->v_proj->prefill(stream, num_tokens, this->attn_norm->output, this->attn_norm->output_scale, this->attn_norm->output_sum, v_cache);
        kv_cache->rotary_embedding->prefill(stream, num_tokens, this->num_attention_heads, this->num_key_value_heads, this->q_proj->output, k_cache, position_ids);

        mha_fwd_kvcache(
            TypeTraits<T>::type_code()==1,
            1,
            num_tokens,
            num_history_tokens+num_tokens,
            num_tokens,
            this->num_attention_heads,
            this->num_key_value_heads,
            this->head_dim,
            this->q_proj->output,
            kv_cache->k_cache,
            kv_cache->v_cache,
            nullptr,
            Mask(nullptr),
            this->attn_output,
            this->softmax_lse,
            this->softmax_lse_accum,
            this->oaccum,
            rsqrtf(float(this->head_dim)),
            true,
            -1,
            -1,
            0,
            stream.stream
        );

        // flash attention and put output to attn_norm->output
        this->o_quant_invoker->invoke(stream, this->attn_output, num_tokens);
        this->o_proj->prefill(stream, num_tokens, this->o_quant_invoker->output, this->o_quant_invoker->output_scale, this->o_quant_invoker->output_sum);
        elementwise_add<T>(stream, num_tokens, this->hidden_size, input, this->o_proj->output, input);
    }

    void decode(const Stream& stream, int32_t num_tokens, int32_t padded_length, T* input, int32_t* position_ids, int32_t* cache_length, const Mask& mask, KVCache<T>* kv_cache) {
        this->attn_norm->prefill(stream, num_tokens, input);
        T *q, *k, *v;
        if (num_tokens > 1) {
            w4a8_qoq_chn_gemm_forward_cuda(
                stream, 
                this->attn_norm->output,
                this->q_proj->weight,
                this->q_proj->s1_scales, 
                this->attn_norm->output_scale,
                this->q_proj->s1_szeros,
                this->attn_norm->output_sum,
                this->v_proj->output,
                num_tokens,
                this->hidden_size,
                num_tokens,
                (this->num_attention_heads + 2 * this->num_key_value_heads) * this->head_dim
            );
            permute(stream, num_tokens, this->num_attention_heads * this->head_dim, this->num_key_value_heads * this->head_dim, this->v_proj->output, this->q_proj->output);
        } else {
            w4a8_qoq_chn_gemm_forward_cuda(
                stream, 
                this->attn_norm->output,
                this->q_proj->weight,
                this->q_proj->s1_scales, 
                this->attn_norm->output_scale,
                this->q_proj->s1_szeros,
                this->attn_norm->output_sum,
                this->q_proj->output,
                num_tokens,
                this->hidden_size,
                num_tokens,
                (this->num_attention_heads + 2 * this->num_key_value_heads) * this->head_dim
            );
        }
        
        q = this->q_proj->output;
        k = q + num_tokens * this->num_attention_heads * this->head_dim;
        v = k + num_tokens * this->num_key_value_heads * this->head_dim;
        // }
        kv_cache->rotary_embedding->prefill(stream, num_tokens, this->num_attention_heads, this->num_key_value_heads, q, k, position_ids);

        copy_to_kvcache(stream, num_tokens, k, v, kv_cache, cache_length);

        mha_fwd_kvcache(
            TypeTraits<T>::type_code()==1,
            1,
            num_tokens,
            padded_length,
            num_tokens,
            this->num_attention_heads,
            this->num_key_value_heads,
            this->head_dim,
            q,
            kv_cache->k_cache,
            kv_cache->v_cache,
            cache_length,
            mask,
            this->attn_output,
            this->softmax_lse,
            this->softmax_lse_accum,
            this->oaccum,
            rsqrtf(float(this->head_dim)),
            true,
            -1,
            -1,
            0,
            stream.stream
        );

        // flash attention and put output to attn_norm->output
        this->o_quant_invoker->invoke(stream, this->attn_output, num_tokens);
        this->o_proj->prefill(stream, num_tokens, this->o_quant_invoker->output, this->o_quant_invoker->output_scale, this->o_quant_invoker->output_sum);
        elementwise_add<T>(stream, num_tokens, this->hidden_size, input, this->o_proj->output, input);
    }
};

