#pragma once
#include "../w4a16_gptq_marlin/w4a16_gptq_marlin_model.cuh"
#include "../eagle3.cuh"
#include "../drafter.cuh"
#include "../w4a16_gptq_marlin/w4a16_gptq_marlin_layer.cuh"


template <typename T>
struct HierEagle3W4A16GMSpecW4A16GMDraftPrefillImpl: Model {

    // eagle3
    int ea_num_iter;
    int ea_topk_per_iter;
    int ea_tree_size;
    int ea_total_tried;
    int ea_draft_hidden_size;
    int ea_draft_intermediate_size;
    int ea_draft_vocab_size;
    int ea_draft_num_key_value_heads;

    Embedding<T>* ea_embedding;
    KVCacheManager<T>* ea_kv_caches;
    LayerEmbed<T>* ea_mid_layer;
    Linear<T> *ea_fc1; // low hidden states
    Linear<T> *ea_fc2; // mid hidden states
    Linear<T> *ea_fc3; // high hidden states
    functions::TopK<T>* ea_topk_func;
    functions::TopK<T>* ea_topk_func_2;
    RMSNorm<T>* ea_norm;
    Linear<T> * ea_draft_lm_head;

    T *ea_prev_low_state, *ea_prev_mid_state, *ea_prev_high_state, *ea_prev_embed;
    T *ea_decode_low_state, *ea_decode_mid_state, *ea_decode_high_state;
    int ea_num_prev, ea_num_history_tokens;
    int32_t *eagle_position_ids, *eagle_cache_length;
    int *eagle_original_length, eagle_padded_length;
    uint64_t *eagle_mask_2d, *ea_tmp_mask_2d;
    T* eagle_logits;
    T* ea_tired_history_val; int32_t* ea_tired_history_pos;
    int32_t* ea_tired_history_parent;
    bool ea_is_first_draft;
    bool ea_load_target_embed;

    int32_t* ea_d2t; // TOOD: index select implement
    uint8_t* ea_t2d;
    

    int32_t *ea_h_best, *ea_d_best;    

    T* ea_tmp_kvcache;

    int32_t* ea_tree_draft_ids, *ea_tree_position_ids, *ea_tree_cache_length, *ea_tree_parent;
    uint64_t* ea_tree_attn_mask;

    // draft & target

    W4A16GPTQMarlinModelImpl<T>* draft_model;
    W4A16GPTQMarlinModelImpl<T>* model;

    // draft args
    int32_t *draft_input;
    int32_t *draft_position_ids, *draft_cache_length;
    int * host_draft_cache_length;
    int draft_padded_length; 
    T* draft_logits;
    bool is_first_draft;
    functions::TopK<T>* topk_func;
    int32_t *draft_tmp;
    int32_t *h_best, *d_best;    
    int num_prev, num_history_tokens;

    // draft mask always nullptr
    uint64_t* draft_mask_2d;   

    // graph
    bool draft_cuda_graph;
    int draft_graphCreated_padding_length;
    int draft_graphCreated_input_length;
    cudaGraph_t draft_graph;
    cudaGraphExec_t draft_graphExec;

    // cascade vars
    int cur_draft_length;
    int min_draft_length;
    // T * draft_tmp_hidden_state;
    T* draft_tmp_low_hidden_state;
    T* draft_tmp_mid_hidden_state;
    T* draft_tmp_high_hidden_state;
    bool draft_model_start; // start from draft model for num_prev == 1

    int32_t* ea_accept_nums;
    int ea_accept_nums_size;
    int cur_ea_accept_nums_size;

    HierEagle3W4A16GMSpecW4A16GMDraftPrefillImpl(
        W4A16GPTQMarlinModelImpl<T>* model,
        int draft_vocab_size,
        int draft_num_hidden_layers,
        int draft_hidden_size,
        int draft_intermediate_size,
        int draft_num_attention_heads,
        int draft_num_key_value_heads,
        int draft_head_dim,
        float draft_rms_norm_eps,
        int draft_group_size,
        // int num_iter,
        int min_draft_length,
        bool draft_cuda_graph,
        // eagle args
        int ea_draft_hidden_size,
        int ea_draft_intermediate_size,
        int ea_draft_num_attention_heads,
        int ea_draft_num_key_value_heads,
        bool ea_load_target_embed,
        int ea_draft_vocab_size,
        int ea_num_iter,
        int ea_topk_per_iter,
        int ea_tree_size,
        bool draft_model_start
    ) {
        this->model = model;
        this->draft_model = new W4A16GPTQMarlinModelImpl<T>(
            0,
            nullptr,
            draft_vocab_size,
            draft_num_hidden_layers,
            draft_hidden_size,
            draft_intermediate_size,
            draft_num_attention_heads,
            draft_num_key_value_heads,
            draft_head_dim,
            draft_rms_norm_eps,
            draft_group_size,
            this->model->chunk_length
        );

        // draft config
        this->draft_mask_2d = 0;
        topk_func = new functions::TopK<T>(model->vocab_size, 1); // greedy sample
        
        this->draft_cuda_graph = draft_cuda_graph;
        this->draft_graphCreated_padding_length = -1;
        this->draft_graphCreated_input_length = -1;
        this->draft_graph = nullptr;
        this->draft_graphExec = nullptr;

        this->min_draft_length = min_draft_length;
        this->draft_model_start = draft_model_start;

        // eagle config
        this->ea_draft_hidden_size = ea_draft_hidden_size;
        this->ea_draft_intermediate_size = ea_draft_intermediate_size;
        this->ea_draft_vocab_size = ea_draft_vocab_size;
        this->ea_draft_num_key_value_heads = ea_draft_num_key_value_heads;
        this->ea_num_iter = ea_num_iter;
        this->ea_topk_per_iter = ea_topk_per_iter;
        this->ea_tree_size = ea_tree_size;
        this->ea_total_tried = ea_topk_per_iter * ea_topk_per_iter * (ea_num_iter-1) +  ea_topk_per_iter;

        // ea model
        ea_embedding = new Embedding<T>(this->draft_model->vocab_size, this->ea_draft_hidden_size);
        ea_kv_caches = new KVCacheManager<T>(1, ea_draft_num_key_value_heads, this->draft_model->head_dim);
        ea_fc1 = new Linear<T>(this->draft_model->hidden_size, this->ea_draft_hidden_size);
        ea_fc2 = new Linear<T>(this->draft_model->hidden_size, this->ea_draft_hidden_size);
        ea_fc3 = new Linear<T>(this->draft_model->hidden_size, this->ea_draft_hidden_size);
        ea_mid_layer = new LayerEmbed<T>(this->ea_draft_hidden_size, this->ea_draft_intermediate_size, ea_draft_num_attention_heads, ea_draft_num_key_value_heads, this->draft_model->head_dim, this->draft_model->rms_norm_eps);
        this->ea_norm = new RMSNorm<T>(this->ea_draft_hidden_size, this->draft_model->rms_norm_eps);
        this->ea_draft_lm_head = new Linear<T>(this->ea_draft_hidden_size, ea_draft_vocab_size);

        ea_topk_func = new functions::TopK<T>(this->ea_draft_vocab_size, ea_topk_per_iter);
        ea_topk_func_2 = new functions::TopK<T>(ea_total_tried, this->ea_tree_size-1); // TODO current topk do not support k > 32

        this->ea_load_target_embed = ea_load_target_embed;

        this->ea_accept_nums_size = 0;
        this->cur_ea_accept_nums_size = 0;
        
    }

    void init_weight_ptr(Memory* memory) {
        ea_embedding->init_weight_ptr(memory);
        ea_fc1->init_weight_ptr(memory);
        ea_fc2->init_weight_ptr(memory);
        ea_fc3->init_weight_ptr(memory);

        ea_mid_layer->init_weight_ptr(memory);
        ea_kv_caches->rotary_embedding = this->draft_model->kv_caches->rotary_embedding;

        ea_norm->init_weight_ptr(memory);
        ea_draft_lm_head->init_weight_ptr(memory);
    }
    

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        // init eagle output
        offset = ea_embedding->init_output_ptr(memory, num_tokens, offset);
        offset = ea_fc1->init_output_ptr(memory, num_tokens, offset);
        offset = ea_fc2->init_output_ptr(memory, num_tokens, offset);
        offset = ea_fc3->init_output_ptr(memory, num_tokens, offset);
        offset = ea_mid_layer->init_output_ptr(memory, num_tokens, offset);

        offset = ea_norm->init_output_ptr(memory, num_tokens, offset);
        offset = ea_draft_lm_head->init_output_ptr(memory, num_tokens, offset);

        offset = memory->allocate((void**)&ea_d2t, offset, this->ea_draft_vocab_size * sizeof(int32_t));
        offset = memory->allocate((void**)&ea_t2d, offset, this->draft_model->vocab_size * sizeof(uint8_t));

        offset = memory->allocate((void**)&eagle_logits, offset, this->ea_topk_per_iter * this->ea_draft_vocab_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_mask_2d, offset, this->ea_topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&ea_tmp_mask_2d, offset, this->ea_topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&ea_tired_history_val, offset, this->ea_total_tried * sizeof(T));
        offset = memory->allocate((void**)&ea_tired_history_pos, offset, this->ea_total_tried * sizeof(int32_t));
        offset = memory->allocate((void**)&ea_tired_history_parent, offset, this->ea_topk_per_iter * (this->ea_num_iter - 1) * sizeof(int32_t));
        cudaMallocHost(&eagle_original_length, sizeof(int32_t));

        offset = ea_topk_func->init_output_ptr(memory, this->ea_topk_per_iter, offset);
        offset = ea_topk_func_2->init_output_ptr(memory, 1, offset);

        offset = memory->allocate((void**)&ea_prev_low_state, offset, num_tokens * this->draft_model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&ea_prev_mid_state, offset, num_tokens * this->draft_model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&ea_prev_high_state, offset, num_tokens * this->draft_model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&ea_decode_low_state, offset, (this->ea_tree_size+1) * this->draft_model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&ea_decode_mid_state, offset, (this->ea_tree_size+1) * this->draft_model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&ea_decode_high_state, offset, (this->ea_tree_size+1) * this->draft_model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&ea_prev_embed, offset, num_tokens * this->ea_draft_hidden_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&eagle_cache_length, offset, sizeof(int32_t));

        offset = memory->allocate((void**)&ea_d_best, offset, 2 * sizeof(int32_t));
        cudaMallocHost(&ea_h_best, 2 * sizeof(int32_t));
        offset = memory->allocate((void**)&ea_tmp_kvcache, offset, 64 * this->draft_model->kv_caches->num_hidden_layers * 2 * this->draft_model->kv_caches->dim * sizeof(T));

        // to allocate ealge draft some states
        offset = memory->allocate((void**)&ea_tree_draft_ids, offset, this->ea_tree_size * sizeof(int32_t));
        offset = memory->allocate((void**)&ea_tree_position_ids, offset, this->ea_tree_size * sizeof(int32_t));
        offset = memory->allocate((void**)&ea_tree_cache_length, offset, sizeof(int32_t));
        offset = memory->allocate((void**)&ea_tree_parent, offset, this->ea_tree_size * sizeof(int32_t));
        offset = memory->allocate((void**)&ea_tree_attn_mask, offset, this->ea_tree_size * sizeof(uint64_t));



        // init draft output
        int64_t lm_head_end = this->draft_model->init_output_ptr(memory, num_tokens, offset);
        offset = lm_head_end;
        
        offset = memory->allocate((void**)&draft_input, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&draft_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&draft_cache_length, offset, sizeof(int32_t));
        cudaMallocHost(&host_draft_cache_length, sizeof(int32_t));

        
        offset = memory->allocate((void**)&draft_logits, offset, 64 * this->draft_model->vocab_size * sizeof(T));
        offset = topk_func->init_output_ptr(memory, 64, offset);
        
        offset = memory->allocate((void**)&draft_tmp, offset, (this->min_draft_length + ea_num_iter + 1)*sizeof(int32_t));
        offset = memory->allocate((void**)&d_best, offset, sizeof(int32_t));
        cudaMallocHost(&h_best, sizeof(int32_t));

        // cascade vars
        // offset = memory->allocate((void**)&draft_tmp_hidden_state, offset, (this->min_draft_length + ea_num_iter + 1) * this->draft_model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&draft_tmp_low_hidden_state, offset, (this->min_draft_length + ea_num_iter + 1) * this->draft_model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&draft_tmp_mid_hidden_state, offset, (this->min_draft_length + ea_num_iter + 1) * this->draft_model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&draft_tmp_high_hidden_state, offset, (this->min_draft_length + ea_num_iter + 1) * this->draft_model->hidden_size * sizeof(T));
        // cudaMallocHost(&host_ea_accept_nums, 1024 * sizeof(int));
        offset = memory->allocate((void**)&ea_accept_nums, offset, 1024 * sizeof(int32_t));
        return offset;
    }

    int init_storage() {

        this->model->init_weight_ptr(this->model->memory);
        // this->init_weight_ptr(this->model->memory);
        this->draft_model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);

        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = init_output_ptr(this->model->memory, this->model->chunk_length, offset);

        int model_kv_size = (this->model->num_hidden_layers*this->model->num_key_value_heads*this->model->head_dim);
        int draft_kv_size = (this->draft_model->num_hidden_layers*this->draft_model->num_key_value_heads*this->draft_model->head_dim);
        int ea_kv_size = this->ea_draft_num_key_value_heads * this->draft_model->head_dim;
        float ratio = float(model_kv_size)/float(model_kv_size + draft_kv_size + ea_kv_size);
        kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        ratio = float(draft_kv_size)/float(draft_kv_size + ea_kv_size);
        kv_cache_offset = this->draft_model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        this->ea_kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, 1.0);
        return min(min(this->draft_model->kv_caches->budget, this->model->kv_caches->budget), this->ea_kv_caches->budget + 1);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 6) == "eagle3") {
            if (name.substr(0, 19) == "eagle3.embed_tokens" && !this->ea_load_target_embed) {
                std::cout << "load draft ea embed" << std::endl;
                ea_embedding->load_to_storage(name, ptr);
            } else if (name.substr(0, 10) == "eagle3.d2t"){
                cudaMemcpy((void*)ea_d2t, ptr, this->ea_draft_vocab_size * sizeof(int32_t), cudaMemcpyHostToDevice);
            } else if (name.substr(0, 10) == "eagle3.t2d") {
                cudaMemcpy((void*)ea_t2d, ptr, this->draft_model->vocab_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
            } else if (name.substr(0, 11) == "eagle3.norm"){
                ea_norm->load_to_storage(name, ptr);
            } else if (name.substr(0, 14) == "eagle3.lm_head"){
                ea_draft_lm_head->load_to_storage(name, ptr);
            } else if (name.substr(0, 10) == "eagle3.fc1") {
                ea_fc1->load_to_storage(name, ptr);
            } else if (name.substr(0, 10) == "eagle3.fc2") {
                ea_fc2->load_to_storage(name, ptr);
            } else if (name.substr(0, 10) == "eagle3.fc3") {
                ea_fc3->load_to_storage(name, ptr);
            } else {
                std::regex layer_regex("eagle3\\.midlayer\\.(.*)");
                std::smatch matches;
                if (std::regex_search(name, matches, layer_regex)) {
                    ea_mid_layer->load_to_storage(matches[1], ptr);
                } else {
                    throw std::invalid_argument("Unsupported name (layer_idx not found): " + name);
                }
            }
        } else if (name.substr(0, 5) == "draft"){
            std::string draft_name = name.substr(6);
            if (this->ea_load_target_embed && draft_name.substr(0, 18) == "model.embed_tokens") {
                std::cout << "load draft target embed" << std::endl;
                ea_embedding->load_to_storage(draft_name, ptr);
            }
            this->draft_model->load_to_storage(draft_name, ptr);
        } else {
            this->model->load_to_storage(name, ptr);
        }
    }



    void eagle_prefill(int num_history_tokens) {
        cudaMemcpy(this->ea_prev_embed + (ea_num_prev - 1) * this->ea_draft_hidden_size, this->ea_embedding->output, this->ea_draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        this->ea_fc1->prefill(calc_stream, ea_num_prev, this->ea_prev_low_state);
        this->ea_fc2->prefill(calc_stream, ea_num_prev, this->ea_prev_mid_state);
        this->ea_fc3->prefill(calc_stream, ea_num_prev, this->ea_prev_high_state);
        elementwise_add3(calc_stream, ea_num_prev, this->ea_draft_hidden_size, this->ea_fc1->output, this->ea_fc2->output, this->ea_fc3->output, this->ea_fc3->output);
        
        T* layer_output = nullptr;
        this->ea_mid_layer->prefill(ea_num_prev, num_history_tokens, this->ea_prev_embed, this->ea_fc3->output, layer_output, this->eagle_position_ids, this->ea_kv_caches->caches[0]);
        layer_output = this->ea_mid_layer->output;

        elementwise_add(calc_stream, ea_num_prev, this->ea_draft_hidden_size, this->ea_fc3->output, layer_output, this->ea_fc3->output);
    }

    void eagle_decode(int32_t* cache_length) {

        this->ea_fc1->prefill(calc_stream, ea_num_prev, this->ea_prev_low_state);
        this->ea_fc2->prefill(calc_stream, ea_num_prev, this->ea_prev_mid_state);
        this->ea_fc3->prefill(calc_stream, ea_num_prev, this->ea_prev_high_state);
        elementwise_add3(calc_stream, ea_num_prev, this->ea_draft_hidden_size, this->ea_fc1->output, this->ea_fc2->output, this->ea_fc3->output, this->ea_fc3->output);
        T* layer_output = nullptr;
        this->ea_mid_layer->decode(ea_num_prev, this->eagle_padded_length, this->ea_prev_embed, this->ea_fc3->output, layer_output, this->eagle_position_ids, cache_length, Mask(nullptr), this->ea_kv_caches->caches[0]);
        layer_output = this->ea_mid_layer->output;
        elementwise_add(calc_stream, ea_num_prev, this->ea_draft_hidden_size, this->ea_fc3->output, layer_output, this->ea_fc3->output);
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->model->prefill(num_tokens, num_history_tokens, input, position_ids, output);
        if (num_history_tokens > 0) {
            this->draft_model->embedding->prefill(calc_stream, this->num_prev, this->draft_input);

            this->ea_embedding->prefill(calc_stream, this->num_prev, this->draft_input);
            cudaMemcpy(this->ea_prev_embed, this->ea_embedding->output+this->ea_draft_hidden_size, (this->num_prev-1) * this->ea_draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

            this->draft_model->prefill_embed_eagle3_states(this->num_prev, this->num_history_tokens, this->draft_model->embedding->output, this->draft_position_ids, (void*)this->draft_logits, this->ea_prev_low_state, this->ea_prev_mid_state, this->ea_prev_high_state);
            
            
            this->ea_embedding->prefill(calc_stream, 1, input);
            this->eagle_prefill(this->ea_num_history_tokens);

            // this->draft_model->
        }
        
        cudaMemcpy(this->draft_input, input, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_position_ids, position_ids, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->num_prev = num_tokens;
        this->num_history_tokens = num_history_tokens;
        this->is_first_draft = true;

        // eagle
        cudaMemcpy(this->eagle_position_ids, this->draft_position_ids, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->ea_num_prev = num_tokens;
        this->ea_num_history_tokens = num_history_tokens;
        this->ea_is_first_draft = true;


        this->ea_accept_nums_size = 0;
        cudaMemcpy(this->ea_accept_nums, &this->ea_accept_nums_size, sizeof(int), cudaMemcpyHostToDevice);
    }

    
    void draft_decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        throw std::runtime_error("Draft decode is not supported");
    }

    void draft_decode_with_graph_control(int32_t d_num_tokens, int32_t d_padded_length, int32_t* d_input, int32_t* d_position_ids, int32_t* d_cache_length, uint64_t* d_mask_2d, void* d_output) {
        if (this->draft_cuda_graph) {
            if (this->draft_graphCreated_padding_length != d_padded_length || this->draft_graphCreated_input_length != d_num_tokens) {
                if (this->draft_graphExec != nullptr) {
                    cudaGraphExecDestroy(this->draft_graphExec);
                    this->draft_graphExec = nullptr;
                }
                if (this->draft_graph != nullptr) {
                    cudaGraphDestroy(this->draft_graph);
                    this->draft_graph = nullptr;
                }
                cudaStreamBeginCapture(calc_stream.stream, cudaStreamCaptureModeGlobal);
                this->draft_model->decode_eagle3_states(d_num_tokens, d_padded_length, d_input, d_position_ids, d_cache_length, d_mask_2d, d_output, this->ea_decode_low_state, this->ea_decode_mid_state, this->ea_decode_high_state);
                cudaStreamEndCapture(calc_stream.stream, &(this->draft_graph));
                cudaGraphInstantiate(&(this->draft_graphExec), this->draft_graph, nullptr, nullptr, 0);
                this->draft_graphCreated_padding_length = d_padded_length;
                this->draft_graphCreated_input_length = d_num_tokens;
            }
            cudaGraphLaunch(this->draft_graphExec, calc_stream.stream);
        } else {
            this->draft_model->decode_eagle3_states(d_num_tokens, d_padded_length, d_input, d_position_ids, d_cache_length, d_mask_2d, d_output, this->ea_decode_low_state, this->ea_decode_mid_state, this->ea_decode_high_state);
        }
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->model->decode(num_tokens, padded_length, input, position_ids, cache_length, nullptr, output);
    }

    
    void draft_with_eagle(int32_t* ea_tree_draft_ids, int32_t* ea_tree_position_ids, int32_t* ea_cache_length, uint64_t* ea_tree_attn_mask, int32_t* ea_tree_parent) {
        cudaMemcpy(this->eagle_original_length, ea_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
        this->eagle_padded_length = (this->eagle_original_length[0] + 256 - 1) / 128 * 128;
        if (this->ea_is_first_draft) {
            // prefill hidden states and embedding have been cpy
            
        } else {
            this->eagle_decode(ea_cache_length);
        }
        
        cudaMemcpy(this->eagle_cache_length, ea_cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->eagle_position_ids, ea_cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        repeat(calc_stream, ea_topk_per_iter, 1, 0, this->eagle_position_ids);

        { // d = 0
            this->ea_norm->prefill(calc_stream, 1, this->ea_fc3->output+(ea_num_prev-1)*this->ea_draft_hidden_size, nullptr);
            this->ea_draft_lm_head->prefill(calc_stream, 1, this->ea_norm->output, this->eagle_logits);
            log_softmax(calc_stream, 1, this->ea_draft_vocab_size, this->eagle_logits);
            this->ea_topk_func->prefill(calc_stream, 1, this->eagle_logits);
            gather_int(calc_stream, this->ea_topk_per_iter, this->ea_d2t, this->ea_topk_func->topk_pos, this->ea_topk_func->topk_pos);
            cudaMemcpy(this->ea_topk_func_2->topk_pos, this->ea_topk_func->topk_pos, ea_topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_topk_func_2->topk_val, this->ea_topk_func->topk_val, ea_topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_tired_history_val, this->ea_topk_func->topk_val, ea_topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_tired_history_pos, this->ea_topk_func->topk_pos, ea_topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            repeat(calc_stream, ea_topk_per_iter, this->ea_draft_hidden_size, ea_num_prev-1, this->ea_fc3->output, this->ea_fc1->output);
            init_tree(calc_stream, ea_topk_per_iter, this->eagle_mask_2d);
        }

        for (int d = 1; d < this->ea_num_iter; ++d) {
            add(calc_stream, 1, this->eagle_cache_length, ea_topk_per_iter);
            this->ea_embedding->prefill(calc_stream, ea_topk_per_iter, this->ea_topk_func_2->topk_pos);
            
            cudaMemcpy(this->ea_fc3->output, this->ea_fc1->output, ea_topk_per_iter * this->ea_draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
            T* layer_output = nullptr;
            this->ea_mid_layer->decode(ea_topk_per_iter, this->eagle_padded_length, this->ea_embedding->output, this->ea_fc3->output, layer_output, this->eagle_position_ids, this->eagle_cache_length, Mask(eagle_mask_2d, ea_topk_per_iter, ea_topk_per_iter * d), this->ea_kv_caches->caches[0]);
            layer_output = this->ea_mid_layer->output;
            this->ea_norm->prefill(calc_stream, ea_topk_per_iter, this->ea_fc3->output, layer_output);

            add(calc_stream, ea_topk_per_iter, this->eagle_position_ids, 1);

            this->ea_draft_lm_head->prefill(calc_stream, ea_topk_per_iter, this->ea_norm->output, this->eagle_logits);
            log_softmax(calc_stream, ea_topk_per_iter, this->ea_draft_vocab_size, this->eagle_logits);
            this->ea_topk_func->prefill(calc_stream, ea_topk_per_iter, this->eagle_logits);
            gather_int(calc_stream, ea_topk_per_iter*ea_topk_per_iter, this->ea_d2t, this->ea_topk_func->topk_pos, this->ea_topk_func->topk_pos);
            
            cumsum(calc_stream, ea_topk_per_iter, ea_topk_per_iter, this->ea_topk_func->topk_val, this->ea_topk_func_2->topk_val);
            cudaMemcpy(this->ea_tired_history_val + ea_topk_per_iter + (d - 1) * ea_topk_per_iter * ea_topk_per_iter, this->ea_topk_func->topk_val, ea_topk_per_iter * ea_topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_tired_history_pos + ea_topk_per_iter + (d - 1) * ea_topk_per_iter * ea_topk_per_iter, this->ea_topk_func->topk_pos, ea_topk_per_iter * ea_topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            this->ea_topk_func_2->prefill(calc_stream, 1, this->ea_topk_func->topk_val, ea_topk_per_iter * ea_topk_per_iter, ea_topk_per_iter);

            cudaMemcpy(this->ea_tmp_mask_2d, this->eagle_mask_2d, ea_topk_per_iter * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            set_parent(calc_stream, ea_topk_per_iter, this->ea_tired_history_parent + (d - 1) * ea_topk_per_iter, this->ea_topk_func_2->topk_pos, 10 + (d - 1) * ea_topk_per_iter * ea_topk_per_iter);
            update_tree(calc_stream, ea_topk_per_iter, ea_topk_per_iter * d, this->eagle_mask_2d, this->ea_tmp_mask_2d, this->ea_topk_func_2->topk_pos);
            remap_hidden(calc_stream, ea_topk_per_iter, this->ea_draft_hidden_size, this->ea_topk_func_2->topk_pos, this->ea_fc3->output, this->ea_fc1->output, ea_topk_per_iter);
            remap_id(calc_stream, ea_topk_per_iter, this->ea_topk_func_2->topk_pos, this->ea_topk_func->topk_pos);
        }

        this->ea_topk_func_2->prefill(calc_stream, 1, this->ea_tired_history_val);

        // build tree
        build_dynamic_tree(calc_stream, this->ea_tree_size, this->eagle_original_length[0], this->ea_topk_per_iter, this->ea_tired_history_parent, this->ea_topk_func_2->topk_pos, ea_tree_position_ids, ea_tree_attn_mask, ea_tree_parent);
        remap_id(calc_stream, this->ea_tree_size-1, this->ea_topk_func_2->topk_pos, this->ea_tired_history_pos, ea_tree_draft_ids + 1);

        this->ea_is_first_draft = false;
    }

    void draft_prefill(int32_t *tree_draft_ids, int32_t *tree_position_ids, int32_t *cache_length) {
        // append tree draft ids to draft input
        cudaMemcpy(this->draft_input+this->num_prev, tree_draft_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_position_ids+this->num_prev, tree_position_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->num_prev += 1;

        this->draft_model->embedding->prefill(calc_stream, this->num_prev, this->draft_input);

        this->ea_embedding->prefill(calc_stream, this->num_prev, this->draft_input);
        cudaMemcpy(this->ea_prev_embed, this->ea_embedding->output+ this->ea_draft_hidden_size, (this->num_prev-1) * this->ea_draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        this->draft_model->prefill_embed_eagle3_states(this->num_prev, this->num_history_tokens, this->draft_model->embedding->output, this->draft_position_ids, (void*)this->draft_logits, this->ea_prev_low_state, this->ea_prev_mid_state, this->ea_prev_high_state);

        // eagle prepare for draft_with_eagle function 
        // ea_is_first_draft is True
        cudaMemcpy(this->eagle_position_ids + (this->ea_num_prev), tree_position_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->ea_num_prev = this->num_prev;
        
        this->topk_func->prefill(calc_stream, 1, this->draft_logits);
        

        // prepare for draft with eagle     
        cudaMemcpy(this->ea_tree_draft_ids, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->ea_tree_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        add(calc_stream, 1, this->ea_tree_cache_length, 1);

        cudaMemcpy(this->draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        add(calc_stream, 1, this->draft_cache_length, 1);
        cudaMemcpy(this->host_draft_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);           

        // draft model has forward one time
        cudaMemcpy(this->draft_tmp, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);

        cudaMemcpy(this->draft_tmp_low_hidden_state, this->ea_prev_low_state + (this->num_prev-1) * this->draft_model->hidden_size, this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_tmp_mid_hidden_state, this->ea_prev_mid_state + (this->num_prev-1) * this->draft_model->hidden_size, this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_tmp_high_hidden_state, this->ea_prev_high_state + (this->num_prev-1) * this->draft_model->hidden_size, this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
        

        this->ea_embedding->prefill(calc_stream, 1, ea_tree_draft_ids);
        this->eagle_prefill(this->ea_num_history_tokens);
    }

    void draft(int32_t *tree_draft_ids, int32_t *tree_position_ids, int32_t *cache_length, uint64_t*, int32_t*) {
        // reset cur draft length
        this->cur_draft_length = 0;
        this->cur_ea_accept_nums_size = 0;
        if (this->is_first_draft) {
            this->cur_draft_length += 1;
            
        } else if (this->num_prev == 2){
            this->draft_model->embedding->prefill(calc_stream, this->num_prev, this->draft_input);

            // new embedding
            this->ea_embedding->prefill(calc_stream, this->num_prev, this->draft_input);
            cudaMemcpy(this->ea_prev_embed + (ea_num_prev)* this->ea_draft_hidden_size, this->ea_embedding->output + this->ea_draft_hidden_size, this->ea_draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

            this->draft_model->decode_embed_eagle3_states(this->num_prev, this->draft_padded_length, this->draft_model->embedding->output, this->draft_position_ids, this->draft_cache_length, nullptr, (void*)this->draft_logits, this->ea_decode_low_state, this->ea_decode_mid_state, this->ea_decode_high_state);


            this->topk_func->prefill(calc_stream, 1, this->draft_logits+(this->draft_model->vocab_size));


            // prepare for the eagle input
            cudaMemcpy(this->ea_prev_low_state + (ea_num_prev)* this->draft_model->hidden_size, this->ea_decode_low_state, this->num_prev * this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_prev_mid_state + (ea_num_prev)* this->draft_model->hidden_size, this->ea_decode_mid_state, this->num_prev * this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_prev_high_state + (ea_num_prev)* this->draft_model->hidden_size, this->ea_decode_high_state, this->num_prev * this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);

            this->ea_embedding->prefill(calc_stream, 1, this->topk_func->topk_pos);
            cudaMemcpy(this->ea_prev_embed + (ea_num_prev+1)* this->ea_draft_hidden_size, this->ea_embedding->output, this->ea_draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
            this->ea_num_prev += this->num_prev;
            

            // prepare for draft with eagle
            cudaMemcpy(this->ea_tree_draft_ids, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_tree_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            
            // draft model has forward one time
            cudaMemcpy(this->draft_tmp, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_tmp_low_hidden_state, this->ea_decode_low_state + (this->num_prev-1) * this->draft_model->hidden_size, this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_tmp_mid_hidden_state, this->ea_decode_mid_state + (this->num_prev-1) * this->draft_model->hidden_size, this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_tmp_high_hidden_state, this->ea_decode_high_state + (this->num_prev-1) * this->draft_model->hidden_size, this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            this->cur_draft_length += 1;
        } else if (this->draft_model_start) {
            // num_prev == 1
            this->draft_model->decode_eagle3_states(this->num_prev, this->draft_padded_length, this->draft_input, this->draft_position_ids, this->draft_cache_length, nullptr, (void*)this->draft_logits, this->ea_decode_low_state, this->ea_decode_mid_state, this->ea_decode_high_state);
            this->topk_func->prefill(calc_stream, 1, this->draft_logits);
            
            // prepare for the eagle input
            // cudaMemcpy(this->ea_prev_hidden_state + (ea_num_prev)*this->draft_model->hidden_size, this->draft_model->norm->output, this->num_prev * this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_prev_low_state + (ea_num_prev)*this->draft_model->hidden_size, this->ea_decode_low_state, this->num_prev * this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_prev_mid_state + (ea_num_prev)*this->draft_model->hidden_size, this->ea_decode_mid_state, this->num_prev * this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_prev_high_state + (ea_num_prev)*this->draft_model->hidden_size, this->ea_decode_high_state, this->num_prev * this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            
            this->ea_embedding->prefill(calc_stream, 1, this->topk_func->topk_pos);
            cudaMemcpy(this->ea_prev_embed + (ea_num_prev)* this->ea_draft_hidden_size, this->ea_embedding->output, this->ea_draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
            this->ea_num_prev += this->num_prev; 

            // prepare for draft with eagle            
            cudaMemcpy(this->ea_tree_draft_ids, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_tree_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);


            // draft model has forward one time
            cudaMemcpy(this->draft_tmp, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_tmp_low_hidden_state, this->ea_decode_low_state + (this->num_prev-1) * this->draft_model->hidden_size, this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_tmp_mid_hidden_state, this->ea_decode_mid_state + (this->num_prev-1) * this->draft_model->hidden_size, this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_tmp_high_hidden_state, this->ea_decode_high_state + (this->num_prev-1) * this->draft_model->hidden_size, this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            this->cur_draft_length += 1;
            
        } else {
            cudaMemcpy(this->ea_tree_draft_ids, tree_draft_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_tree_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        }


        
        while (this->cur_draft_length < min_draft_length){

            // eagle draft
            this->draft_with_eagle(
                this->ea_tree_draft_ids,
                this->ea_tree_position_ids,
                this->ea_tree_cache_length,
                this->ea_tree_attn_mask,
                this->ea_tree_parent
            );

            
            cudaMemcpy(this->draft_cache_length, this->ea_tree_cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_cache_length, this->ea_tree_size);
            cudaMemcpy(this->host_draft_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
            this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;;

            this->draft_decode_with_graph_control(
                this->ea_tree_size,
                this->draft_padded_length,
                this->ea_tree_draft_ids,
                this->ea_tree_position_ids,
                this->draft_cache_length,
                this->ea_tree_attn_mask,
                (void*) this->draft_logits
            );
            this->topk_func->prefill(calc_stream, this->ea_tree_size, this->draft_logits);

            
            this->draft_verify(
                this->ea_tree_size,
                this->ea_tree_draft_ids,
                this->topk_func->topk_pos,
                this->ea_tree_position_ids,
                this->ea_tree_cache_length,
                this->ea_tree_attn_mask,
                this->ea_tree_parent
            );
            
            cudaMemcpy(this->ea_accept_nums + (this->ea_accept_nums_size+this->cur_ea_accept_nums_size+1), this->ea_d_best, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            this->cur_ea_accept_nums_size += 1;
            

            // accept return to ea_h_best[0]
            cudaMemcpy(this->ea_tree_draft_ids, this->ea_tree_draft_ids + (this->ea_h_best[0]-1), sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->ea_tree_cache_length, this->ea_h_best[0]);
            
        }

        

        cudaMemcpy(tree_draft_ids + 1, this->draft_tmp, this->cur_draft_length*sizeof(int32_t), cudaMemcpyDeviceToDevice);
        make_arange(calc_stream, this->cur_draft_length+1, cache_length, tree_position_ids);
        this->is_first_draft = false;
    }
    
    int draft_verify(int32_t ea_num_tokens, int32_t* ea_pred, int32_t* ea_gt, int32_t* ea_position_ids, int32_t* ea_cache_length, uint64_t* ea_mask_2d, int32_t* ea_tree_parent) {
        verify_draft(calc_stream, ea_num_tokens, ea_pred, ea_gt, ea_position_ids, ea_cache_length, ea_mask_2d, ea_tree_parent, this->ea_d_best);
    
        cudaMemcpyAsync(this->ea_h_best, this->ea_d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);

        this->ea_num_prev = ea_h_best[0];
        remap_hidden(calc_stream, this->ea_num_prev, this->draft_model->hidden_size, ea_pred, this->ea_decode_low_state, this->ea_prev_low_state);
        remap_hidden(calc_stream, this->ea_num_prev, this->draft_model->hidden_size, ea_pred, this->ea_decode_mid_state, this->ea_prev_mid_state);
        remap_hidden(calc_stream, this->ea_num_prev, this->draft_model->hidden_size, ea_pred, this->ea_decode_high_state, this->ea_prev_high_state);

        fix_kv_cache(calc_stream, ea_h_best[0], this->draft_model->kv_caches->num_hidden_layers * 2, this->draft_model->kv_caches->dim, ea_pred, ea_gt, ea_cache_length, this->draft_model->kv_caches->d_flat_caches, this->ea_tmp_kvcache);

        this->ea_embedding->prefill(calc_stream, this->ea_num_prev, ea_pred);
        cudaMemcpy(this->ea_prev_embed, this->ea_embedding->output, this->ea_num_prev * this->ea_draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        make_arange(calc_stream, this->ea_num_prev, ea_cache_length, this->eagle_position_ids);

        // cudaMemcpy(this->draft_tmp_hidden_state + (this->cur_draft_length*this->draft_model->hidden_size), this->ea_prev_hidden_state, this->ea_num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_tmp_low_hidden_state + (this->cur_draft_length*this->draft_model->hidden_size), this->ea_prev_low_state, this->ea_num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_tmp_mid_hidden_state + (this->cur_draft_length*this->draft_model->hidden_size), this->ea_prev_mid_state, this->ea_num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_tmp_high_hidden_state + (this->cur_draft_length*this->draft_model->hidden_size), this->ea_prev_high_state, this->ea_num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        cudaMemcpy(this->draft_tmp + this->cur_draft_length, ea_pred, this->ea_num_prev * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->cur_draft_length += this->ea_num_prev;

        return ea_h_best[0];
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* attn_mask, int32_t* tree_parent) { 
        verify_seq_draft(calc_stream, num_tokens, pred, gt, (uint16_t*)attn_mask, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);
        
        if (h_best[0]>(this->cur_draft_length+1)) {
            h_best[0] = this->cur_draft_length+1;
        }
        
        if (h_best[0]==(this->cur_draft_length+1)) {
            // full accept   
            this->num_prev = 2;
            cudaMemcpy(this->draft_input, gt + (this->cur_draft_length-1), 2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_cache_length, this->h_best[0]+1);
            make_arange(calc_stream, 2, cache_length, this->draft_position_ids);
            add(calc_stream, 2, this->draft_position_ids, this->cur_draft_length);
            cudaMemcpy(this->host_draft_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
            this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;
        } else {
            this->num_prev = 1;
            cudaMemcpy(this->draft_input, gt + (this->h_best[0]-1), sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_cache_length, this->h_best[0]+1);
            cudaMemcpy(this->draft_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_position_ids, this->h_best[0]);
            cudaMemcpy(this->host_draft_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
            this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;

            // adapt eagle draft ptr
            // conidtion 1: eagle last start postion is larger than accept position
            if (host_draft_cache_length[0] > this->eagle_original_length[0] + 1) {
                this->ea_num_prev = host_draft_cache_length[0] - (this->eagle_original_length[0] + 1);
                // keep ea_prev_hidden_state and update ea_prev_embed
                this->ea_embedding->prefill(calc_stream, 1, this->draft_input);
                cudaMemcpy(this->ea_prev_embed+(this->ea_num_prev-1), this->ea_embedding->output, this->ea_draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
            } else {
                // condition 2: eagle last start position is less than accept position
                // index from the kepted draft model hidden state
                // cudaMemcpy(this->ea_prev_hidden_state, this->draft_tmp_hidden_state + (h_best[0]-1) *this->draft_model->hidden_size, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
                cudaMemcpy(this->ea_prev_low_state, this->draft_tmp_low_hidden_state + (h_best[0]-1) *this->draft_model->hidden_size, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
                cudaMemcpy(this->ea_prev_mid_state, this->draft_tmp_mid_hidden_state + (h_best[0]-1) *this->draft_model->hidden_size, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
                cudaMemcpy(this->ea_prev_high_state, this->draft_tmp_high_hidden_state + (h_best[0]-1) *this->draft_model->hidden_size, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
                this->ea_num_prev = 1;
                this->ea_embedding->prefill(calc_stream, 1, this->draft_input);
                cudaMemcpy(this->ea_prev_embed, this->ea_embedding->output, this->ea_draft_hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
                cudaMemcpy(this->eagle_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
                add(calc_stream, 1, this->eagle_position_ids, this->h_best[0]-1);
            }
            
            
        }


        this->ea_accept_nums_size += this->cur_ea_accept_nums_size;
        add(calc_stream, 1, this->ea_accept_nums, this->cur_ea_accept_nums_size);
        cudaMemcpy(tree_parent, this->ea_accept_nums, (this->ea_accept_nums_size+1) * sizeof(int32_t), cudaMemcpyDeviceToDevice);


        
        return h_best[0];

    }
};