#pragma once
#include "../w4a16_gptq_marlin/w4a16_gptq_marlin_model.cuh"
#include "../eagle3.cuh"

template<typename T>
struct Eagle3ImplBaseW4A16GPTQMarlinDraftPrefill : Model {
    int num_iter;
    int topk_per_iter;
    int tree_size;
    int total_tried;
    int draft_hidden_size;
    int draft_intermediate_size;
    int draft_vocab_size;

    Embedding<T>* embedding;
    W4A16GPTQMarlinModelImpl<T>* model;
    KVCacheManager<T>* kv_caches;
    LayerEmbed<T>* mid_layer;
    Linear<T> *fc1; // low hidden states
    Linear<T> *fc2; // mid hidden states
    Linear<T> *fc3; // high hidden states
    functions::TopK<T>* topk_func;
    functions::TopK<T>* topk_func_2;
    RMSNorm<T>* norm;
    Linear<T> * draft_lm_head;

    int32_t* d2t; // TOOD: index select implement
    uint8_t* t2d;

    T *prev_low_state, *prev_mid_state, *prev_high_state, *prev_embed;
    T *decode_low_state, *decode_mid_state, *decode_high_state;
    int num_prev, num_history_tokens;
    int32_t *eagle_position_ids, *eagle_cache_length;
    int *eagle_original_length, eagle_padded_length;
    uint64_t *eagle_mask_2d, *tmp_mask_2d;
    T* eagle_logits;
    T* tired_history_val; int32_t* tired_history_pos;
    int32_t* tired_history_parent;
    bool is_first_draft;
    bool load_target_embed;

    int32_t *h_best, *d_best;    

    T* tmp_kvcache;

    Eagle3ImplBaseW4A16GPTQMarlinDraftPrefill(
        W4A16GPTQMarlinModelImpl<T>* model,
        int draft_hidden_size,
        int draft_intermediate_size,
        int draft_num_attention_heads,
        int draft_num_key_value_heads,
        bool load_target_embed,
        int draft_vocab_size,
        int num_iter,
        int topk_per_iter,
        int tree_size
    ) {
        this->model = model;
        this->draft_hidden_size = draft_hidden_size;
        this->draft_intermediate_size = draft_intermediate_size;
        this->draft_vocab_size = draft_vocab_size;
        this->num_iter = num_iter;
        this->topk_per_iter = topk_per_iter;
        this->tree_size = tree_size;
        this->total_tried = topk_per_iter * topk_per_iter * (num_iter - 1) + topk_per_iter;

        embedding = new Embedding<T>(this->model->vocab_size, this->draft_hidden_size);
        kv_caches = new KVCacheManager<T>(1, draft_num_key_value_heads, this->model->head_dim);
        fc1 = new Linear<T>(this->model->hidden_size, this->draft_hidden_size);
        fc2 = new Linear<T>(this->model->hidden_size, this->draft_hidden_size);
        fc3 = new Linear<T>(this->model->hidden_size, this->draft_hidden_size);
        mid_layer = new LayerEmbed<T>(this->draft_hidden_size, this->draft_intermediate_size, draft_num_attention_heads, draft_num_key_value_heads, this->model->head_dim, this->model->rms_norm_eps);
        this->norm = new RMSNorm<T>(this->draft_hidden_size, this->model->rms_norm_eps);
        this->draft_lm_head = new Linear<T>(this->draft_hidden_size, draft_vocab_size);

        topk_func = new functions::TopK<T>(draft_vocab_size, topk_per_iter);
        topk_func_2 = new functions::TopK<T>(total_tried, this->tree_size-1); // TODO current topk do not support k > 32

        this->load_target_embed = load_target_embed;
    }

    void init_weight_ptr(Memory* memory) {
        embedding->init_weight_ptr(memory);
        fc1->init_weight_ptr(memory);
        fc2->init_weight_ptr(memory);
        fc3->init_weight_ptr(memory);

        mid_layer->init_weight_ptr(memory);
        // mid_layer->attn->attn_norm = new Skip<T>(this->model->hidden_size);
        kv_caches->rotary_embedding = this->model->kv_caches->rotary_embedding;

        norm->init_weight_ptr(memory);
        draft_lm_head->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        offset = embedding->init_output_ptr(memory, num_tokens, offset);
        offset = fc1->init_output_ptr(memory, num_tokens, offset);
        offset = fc2->init_output_ptr(memory, num_tokens, offset);
        offset = fc3->init_output_ptr(memory, num_tokens, offset);
        offset = mid_layer->init_output_ptr(memory, num_tokens, offset);

        offset = norm->init_output_ptr(memory, num_tokens, offset);
        offset = draft_lm_head->init_output_ptr(memory, num_tokens, offset);

        offset = memory->allocate((void**)&d2t, offset, this->draft_vocab_size * sizeof(int32_t));
        offset = memory->allocate((void**)&t2d, offset, this->model->vocab_size * sizeof(uint8_t));

        offset = memory->allocate((void**)&eagle_logits, offset, this->topk_per_iter * this->draft_vocab_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tmp_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tired_history_val, offset, this->total_tried * sizeof(T));
        offset = memory->allocate((void**)&tired_history_pos, offset, this->total_tried * sizeof(int32_t));
        offset = memory->allocate((void**)&tired_history_parent, offset, this->topk_per_iter * (this->num_iter - 1) * sizeof(int32_t));
        cudaMallocHost(&eagle_original_length, sizeof(int32_t));

        offset = topk_func->init_output_ptr(memory, this->topk_per_iter, offset);
        offset = topk_func_2->init_output_ptr(memory, 1, offset);

        offset = memory->allocate((void**)&prev_low_state, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&prev_mid_state, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&prev_high_state, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&decode_low_state, offset, (this->tree_size+1) * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&decode_mid_state, offset, (this->tree_size+1) * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&decode_high_state, offset, (this->tree_size+1) * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&prev_embed, offset, num_tokens * this->draft_hidden_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&eagle_cache_length, offset, sizeof(int32_t));

        offset = memory->allocate((void**)&d_best, offset, 2 * sizeof(int32_t));
        cudaMallocHost(&h_best, 2 * sizeof(int32_t));
        offset = memory->allocate((void**)&tmp_kvcache, offset, 64 * this->model->kv_caches->num_hidden_layers * 2 * this->model->kv_caches->dim * sizeof(T));
        return offset;
    }

    int init_storage() {
        this->model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);
        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = this->init_output_ptr(this->model->memory, this->model->chunk_length, offset);
        float ratio = float(this->model->num_hidden_layers) / (this->model->num_hidden_layers + 1);
        kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        kv_caches->init_output_ptr(this->model->memory, kv_cache_offset);
        return min(kv_caches->budget + 1, this->model->kv_caches->budget);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 6) == "eagle3") {
            if (name.substr(0, 19) == "eagle3.embed_tokens" && !this->load_target_embed) {
                std::cout << "load draft embed" << std::endl;
                embedding->load_to_storage(name, ptr);
            } else if (name.substr(0, 10) == "eagle3.d2t"){
                cudaMemcpy((void*)d2t, ptr, this->draft_vocab_size * sizeof(int32_t), cudaMemcpyHostToDevice);
            } else if (name.substr(0, 10) == "eagle3.t2d") {
                cudaMemcpy((void*)t2d, ptr, this->model->vocab_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
            } else if (name.substr(0, 11) == "eagle3.norm"){
                norm->load_to_storage(name, ptr);
            } else if (name.substr(0, 14) == "eagle3.lm_head"){
                draft_lm_head->load_to_storage(name, ptr);
            } else if (name.substr(0, 10) == "eagle3.fc1") {
                fc1->load_to_storage(name, ptr);
            } else if (name.substr(0, 10) == "eagle3.fc2") {
                fc2->load_to_storage(name, ptr);
            } else if (name.substr(0, 10) == "eagle3.fc3") {
                fc3->load_to_storage(name, ptr);
            } else {
                std::regex layer_regex("eagle3\\.midlayer\\.(.*)");
                std::smatch matches;
                if (std::regex_search(name, matches, layer_regex)) {
                    mid_layer->load_to_storage(matches[1], ptr);
                } else {
                    throw std::invalid_argument("Unsupported name (layer_idx not found): " + name);
                }
            }
        } else {
            if (this->load_target_embed && name.substr(0, 18) == "model.embed_tokens") {
                std::cout << "load target embed" << std::endl;
                embedding->load_to_storage(name, ptr);
            }
            this->model->load_to_storage(name, ptr);
        }
    }

    void eagle_prefill(int num_history_tokens) {
        cudaMemcpy(this->prev_embed + (num_prev - 1) * this->draft_hidden_size, this->embedding->output, this->draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        this->fc1->prefill(calc_stream, num_prev, this->prev_low_state);
        this->fc2->prefill(calc_stream, num_prev, this->prev_mid_state);
        this->fc3->prefill(calc_stream, num_prev, this->prev_high_state);
        elementwise_add3(calc_stream, num_prev, this->draft_hidden_size, this->fc1->output, this->fc2->output, this->fc3->output, this->fc3->output);

        T* layer_output = nullptr;
        this->mid_layer->prefill(num_prev, num_history_tokens, this->prev_embed, this->fc3->output, layer_output, this->eagle_position_ids, this->kv_caches->caches[0]);
        layer_output = this->mid_layer->output;

        elementwise_add(calc_stream, num_prev, this->draft_hidden_size, this->fc3->output, layer_output, this->fc3->output);
    }

    void eagle_decode(int32_t* cache_length) {
        this->fc1->prefill(calc_stream, num_prev, this->prev_low_state);
        this->fc2->prefill(calc_stream, num_prev, this->prev_mid_state);
        this->fc3->prefill(calc_stream, num_prev, this->prev_high_state);
        elementwise_add3(calc_stream, num_prev, this->draft_hidden_size, this->fc1->output, this->fc2->output, this->fc3->output, this->fc3->output);
        T* layer_output = nullptr;
        this->mid_layer->decode(num_prev, this->eagle_padded_length, this->prev_embed, this->fc3->output, layer_output, this->eagle_position_ids, cache_length, Mask(nullptr), this->kv_caches->caches[0]);
        layer_output = this->mid_layer->output;
        elementwise_add(calc_stream, num_prev, this->draft_hidden_size, this->fc3->output, layer_output, this->fc3->output);
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->model->embedding->prefill(calc_stream, num_tokens, input);
        
        if (num_history_tokens > 0) {
            this->eagle_prefill(this->num_history_tokens);
        }
        this->embedding->prefill(calc_stream, num_tokens, input);
        cudaMemcpy(this->prev_embed, this->embedding->output + this->draft_hidden_size, (num_tokens - 1) * this->draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        this->model->prefill_embed_eagle3_states(num_tokens, num_history_tokens, this->model->embedding->output, position_ids, output, this->prev_low_state, this->prev_mid_state, this->prev_high_state);
        // this->prev_hidden_state = this->model->norm->output;
        cudaMemcpy(this->eagle_position_ids, position_ids, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->num_prev = num_tokens;

        this->num_history_tokens = num_history_tokens;
        this->is_first_draft = true;
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->model->decode_eagle3_states(num_tokens, padded_length, input, position_ids, cache_length, mask_2d, output, (void*)this->decode_low_state, (void*)this->decode_mid_state, (void*)this->decode_high_state);
    }
    
    
    void draft_prefill(int32_t *tree_draft_ids, int32_t *tree_position_ids, int32_t *cache_length){
        this->embedding->prefill(calc_stream, 1, tree_draft_ids);
        this->eagle_prefill(this->num_history_tokens);
    }

    void draft(int32_t* tree_draft_ids, int32_t* tree_position_ids, int32_t* cache_length, uint64_t* tree_attn_mask, int32_t* tree_parent) {
        cudaMemcpy(this->eagle_original_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
        this->eagle_padded_length = (this->eagle_original_length[0] + 256 - 1) / 128 * 128;


        if (this->is_first_draft) {

        } else {
            this->eagle_decode(cache_length);
        }
        cudaMemcpy(this->eagle_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->eagle_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        repeat(calc_stream, topk_per_iter, 1, 0, this->eagle_position_ids);

        { // d = 0
            this->norm->prefill(calc_stream, 1, this->fc3->output + (num_prev - 1) * this->draft_hidden_size, nullptr);
            this->draft_lm_head->prefill(calc_stream, 1, this->norm->output, this->eagle_logits);
            log_softmax(calc_stream, 1, this->draft_vocab_size, this->eagle_logits);
            this->topk_func->prefill(calc_stream, 1, this->eagle_logits);
            gather_int(calc_stream, this->topk_per_iter, this->d2t, this->topk_func->topk_pos, this->topk_func->topk_pos);
            cudaMemcpy(this->topk_func_2->topk_pos, this->topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->topk_func_2->topk_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tired_history_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tired_history_pos, this->topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            repeat(calc_stream, topk_per_iter, this->draft_hidden_size, num_prev-1, this->fc3->output, this->fc1->output);
            init_tree(calc_stream, topk_per_iter, this->eagle_mask_2d);
        }
        for (int d = 1; d < this->num_iter; ++d) {
            add(calc_stream, 1, this->eagle_cache_length, topk_per_iter);
            this->embedding->prefill(calc_stream, topk_per_iter, this->topk_func_2->topk_pos);
            cudaMemcpy(this->fc3->output, this->fc1->output, topk_per_iter * this->draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

            T* layer_output = nullptr;
            this->mid_layer->decode(topk_per_iter, this->eagle_padded_length, this->embedding->output, this->fc3->output, layer_output, this->eagle_position_ids, this->eagle_cache_length, Mask(eagle_mask_2d, topk_per_iter, topk_per_iter * d), this->kv_caches->caches[0]);
            layer_output = this->mid_layer->output;

            // elementwise_add(calc_stream, topk_per_iter, this->model->hidden_size, this->fc3->output, layer_output, this->fc3->output);
            this->norm->prefill(calc_stream, topk_per_iter, this->fc3->output, layer_output);
            add(calc_stream, topk_per_iter, this->eagle_position_ids, 1);

            this->draft_lm_head->prefill(calc_stream, topk_per_iter, this->norm->output, this->eagle_logits);
            log_softmax(calc_stream, topk_per_iter, this->draft_vocab_size, this->eagle_logits);
            this->topk_func->prefill(calc_stream, topk_per_iter, this->eagle_logits);
            gather_int(calc_stream, topk_per_iter*topk_per_iter, this->d2t, this->topk_func->topk_pos, this->topk_func->topk_pos);

            cumsum(calc_stream, topk_per_iter, topk_per_iter, this->topk_func->topk_val, this->topk_func_2->topk_val);
            cudaMemcpy(this->tired_history_val + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_val, topk_per_iter * topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tired_history_pos + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_pos, topk_per_iter * topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            this->topk_func_2->prefill(calc_stream, 1, this->topk_func->topk_val, topk_per_iter * topk_per_iter, topk_per_iter);

            cudaMemcpy(this->tmp_mask_2d, this->eagle_mask_2d, topk_per_iter * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            set_parent(calc_stream, topk_per_iter, this->tired_history_parent + (d - 1) * topk_per_iter, this->topk_func_2->topk_pos, 10 + (d - 1) * topk_per_iter * topk_per_iter);
            update_tree(calc_stream, topk_per_iter, topk_per_iter * d, this->eagle_mask_2d, this->tmp_mask_2d, this->topk_func_2->topk_pos);
            remap_hidden(calc_stream, topk_per_iter, this->draft_hidden_size, this->topk_func_2->topk_pos, this->fc3->output, this->fc1->output, topk_per_iter);
            remap_id(calc_stream, topk_per_iter, this->topk_func_2->topk_pos, this->topk_func->topk_pos);
        }

        this->topk_func_2->prefill(calc_stream, 1, this->tired_history_val);

        // build tree
        build_dynamic_tree(calc_stream, this->tree_size, this->eagle_original_length[0], this->topk_per_iter, this->tired_history_parent, this->topk_func_2->topk_pos, tree_position_ids, tree_attn_mask, tree_parent);
        remap_id(calc_stream, this->tree_size-1, this->topk_func_2->topk_pos, this->tired_history_pos, tree_draft_ids + 1);

        this->is_first_draft = false;
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, int32_t* tree_parent) {
        verify_draft(calc_stream, num_tokens, pred, gt, position_ids, cache_length, mask_2d, tree_parent, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);

        this->num_prev = h_best[0];
        remap_hidden(calc_stream, this->num_prev, this->model->hidden_size, pred, this->decode_low_state, this->prev_low_state);
        remap_hidden(calc_stream, this->num_prev, this->model->hidden_size, pred, this->decode_mid_state, this->prev_mid_state);
        remap_hidden(calc_stream, this->num_prev, this->model->hidden_size, pred, this->decode_high_state, this->prev_high_state);

        fix_kv_cache(calc_stream, h_best[0], this->model->kv_caches->num_hidden_layers * 2, this->model->kv_caches->dim, pred, gt, cache_length, this->model->kv_caches->d_flat_caches, this->tmp_kvcache);

        this->embedding->prefill(calc_stream, this->num_prev, pred);
        cudaMemcpy(this->prev_embed, this->embedding->output, this->num_prev * this->draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        make_arange(calc_stream, this->num_prev, cache_length, this->eagle_position_ids);

        return h_best[0];
    }
};