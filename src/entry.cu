#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

#include "utils.cuh"
#include "trait.cuh"
// base model
#include "model/model.cuh"
#include "model/w8a8/w8a8_model.cuh"
#include "model/w4a16_gptq_marlin/w4a16_gptq_marlin_model.cuh"
#include "model/w4a8_qqq/w4a8_qqq_model.cuh"
#include "model/w4a8_qoq_chn/w4a8_qoq_chn_model.cuh"
#include "model/w4a8_qoq_group/w4a8_qoq_group_model.cuh"


// eagle
#include "model/eagle.cuh"
#include "model/eagle_base_quant/eagle_base_w8a8.cuh"
#include "model/eagle_base_quant/eagle_base_w4a16_gptq_marlin.cuh"
#include "model/eagle_base_quant/eagle_base_w4a16_gptq_marlin_rot.cuh"
#include "model/eagle_base_quant/eagle_base_w4a8_qqq_rot.cuh"
#include "model/eagle_base_quant/eagle_base_w4a8_qoq_chn_rot.cuh"
#include "model/eagle_base_quant/eagle_base_w4a8_qoq_group_rot.cuh"

// eagle latency
#include "model/eagle_base_quant/eagle_base_w4a16_gptq_marlin_dprefill.cuh"
#include "model/eagle_base_quant/eagle_base_w4a16_gptq_marlin_rot_dprefill.cuh"

// spec model
#include "model/spec_quant/w4a16_gm_spec_w4a16_gm.cuh"
#include "model/spec_quant/w4a16_gm_spec_w4a16_gm_dprefill.cuh"

// hier
#include "model/hier_spec_quant/hier_ea_w4a16_gm_spec_w4a16_gm.cuh"
#include "model/hier_spec_quant/hier_ea_w4a16_gm_spec_w4a16_gm_dprefill.cuh"
#include "model/hier_spec_quant/hier_ea_w4a16_gm_rot_spec_w4a16_gm.cuh"
#include "model/hier_spec_quant/hier_ea_w4a16_gm_rot_spec_w4a16_gm_dprefill.cuh"

// eagle3
#include "model/eagle3.cuh"
#include "model/eagle3_base_quant/eagle3_base_w4a16_gptq_marlin.cuh"
#include "model/eagle3_base_quant/eagle3_base_w4a16_gptq_marlin_dprefill.cuh"

// hier ea3
#include "model/hier_eagle3_spec_quant/hier_ea3_w4a16_gm_spec_w4a16_gm.cuh"
#include "model/hier_eagle3_spec_quant/hier_ea3_w4a16_gm_spec_w4a16_gm_dprefill.cuh"

// gptq marlin convert entry
#include "qgemm/gptq_marlin/gptq_marlin_repack.cuh"


#define DTYPE_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND == 0) {                              \
      using elem_type = __half;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = __nv_bfloat16; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

Model* model;

void init_base_model(
    int64_t memory_limit,
    std::uintptr_t memory_pool,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int torch_dtype,
    int chunk_length
) {
    init_resources();

    DTYPE_SWITCH(torch_dtype, [&] {
        model = new ModelImpl<elem_type>(
            memory_limit,
            reinterpret_cast<void*>(memory_pool),
            vocab_size,
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            chunk_length
        );
    });

}

void init_w8a8_base_model(
    int64_t memory_limit,
    std::uintptr_t memory_pool,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int torch_dtype,
    int chunk_length
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    init_resources();

    model = new W8A8ModelImpl<half>(
        memory_limit,
        reinterpret_cast<void*>(memory_pool),
        vocab_size,
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps,
        chunk_length
    );

}

void init_w4a16_gptq_marlin_base_model(
    int64_t memory_limit,
    std::uintptr_t memory_pool,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int group_size,
    int torch_dtype,
    int chunk_length
) {
    init_resources();
    DTYPE_SWITCH(torch_dtype, [&] {
        model = new W4A16GPTQMarlinModelImpl<elem_type>(
            memory_limit,
            reinterpret_cast<void*>(memory_pool),
            vocab_size,
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            group_size,
            chunk_length
        );
    });
}

void init_w4a8_qqq_base_model(
    int64_t memory_limit,
    std::uintptr_t memory_pool,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int group_size,
    int torch_dtype,
    int chunk_length
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    init_resources();

    model = new W4A8QQQModelImpl<half>(
        memory_limit,
        reinterpret_cast<void*>(memory_pool),
        vocab_size,
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps,
        group_size,
        chunk_length
    );

}

void init_w4a8_qoq_chn_base_model(
    int64_t memory_limit,
    std::uintptr_t memory_pool,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int torch_dtype,
    int chunk_length
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    init_resources();

    model = new W4A8QoQChnModelImpl<half>(
        memory_limit,
        reinterpret_cast<void*>(memory_pool),
        vocab_size,
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps,
        chunk_length
    );
}

void init_w4a8_qoq_group_base_model(
    int64_t memory_limit,
    std::uintptr_t memory_pool,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int torch_dtype,
    int chunk_length
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    init_resources();

    model = new W4A8QoQGroupModelImpl<half>(
        memory_limit,
        reinterpret_cast<void*>(memory_pool),
        vocab_size,
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps,
        chunk_length
    );
}





// eagle model
void init_eagle_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    DTYPE_SWITCH(torch_dtype, [&] {
        model = new EagleImpl<elem_type>(
            (ModelImpl<elem_type>*)model,
            num_layers,
            num_iter,
            topk_per_iter,
            tree_size
        );
    });
}
void init_eagle_w8a8_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new EagleImplBaseW8A8<half>(
        (W8A8ModelImpl<half>*)model,
        num_layers,
        num_iter,
        topk_per_iter,
        tree_size
    );
}

void init_eagle_w4a16_gptq_marlin_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new EagleImplBaseW4A16GPTQMarlin<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        num_layers,
        num_iter,
        topk_per_iter,
        tree_size
    );
}

void init_eagle_w4a16_gptq_marlin_rot_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new EagleImplBaseW4A16GPTQMarlinRot<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        num_layers,
        num_iter,
        topk_per_iter,
        tree_size
    );
}

void init_eagle_w4a8_qqq_rot_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new  EagleImplBaseW4A8QQQRot<half>(
        (W4A8QQQModelImpl<half>*)model,
        num_layers,
        num_iter,
        topk_per_iter,
        tree_size
    );
}

void init_eagle_w4a8_qoq_chn_rot_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new EagleImplBaseW4A8QoQChnRot<half>(
        (W4A8QoQChnModelImpl<half>*)model,
        num_layers,
        num_iter,
        topk_per_iter,
        tree_size
    );
}


void init_eagle_w4a8_qoq_group_rot_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new EagleImplBaseW4A8QoQGroupRot<half>(
        (W4A8QoQGroupModelImpl<half>*)model,
        num_layers,
        num_iter,
        topk_per_iter,
        tree_size
    );
}


// eagle model

void init_eagle_w4a16_gptq_marlin_dprefill_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new EagleImplBaseW4A16GPTQMarlinDraftPrefill<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        num_layers,
        num_iter,
        topk_per_iter,
        tree_size
    );
}

void init_eagle_w4a16_gptq_marlin_rot_dprefill_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new EagleImplBaseW4A16GPTQMarlinRotDraftPrefill<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        num_layers,
        num_iter,
        topk_per_iter,
        tree_size
    );
}



// spec model
void init_w4a16_gm_spec_w4a16_gm_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int draft_group_size,
    int num_iter,
    bool draft_cuda_graph,
    int torch_dtype
) {
    // TODO: different type 
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for spec model");
    }
    // DTYPE_SWITCH(torch_dtype, [&] {
    model = new W4A16GMSpecW4A16GMImpl<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        draft_vocab_size,
        draft_num_hidden_layers,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        draft_head_dim,
        draft_rms_norm_eps,
        draft_group_size,
        num_iter, 
        draft_cuda_graph
    );
    // });
}


void init_w4a16_gm_spec_w4a16_gm_dprefill_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int draft_group_size,
    int num_iter,
    bool draft_cuda_graph,
    int torch_dtype
) {
    // TODO: different type 
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for spec model");
    }
    // DTYPE_SWITCH(torch_dtype, [&] {
    model = new W4A16GMSpecW4A16GMDraftPrefillImpl<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        draft_vocab_size,
        draft_num_hidden_layers,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        draft_head_dim,
        draft_rms_norm_eps,
        draft_group_size,
        num_iter, 
        draft_cuda_graph
    );
    // });
}





// hier spec model
void init_hier_eagle_w4a16_gm_spec_w4a16_gm_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int draft_group_size,
    int min_draft_length,
    bool draft_cuda_graph,
    int ea_num_layers,
    int ea_num_iter,
    int ea_topk_per_iter,
    int ea_tree_size,
    bool draft_model_start,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new HierEagleW4A16GMSpecW4A16GMImpl<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        draft_vocab_size,
        draft_num_hidden_layers,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        draft_head_dim,
        draft_rms_norm_eps,
        draft_group_size,
        min_draft_length,
        draft_cuda_graph,
        ea_num_layers,
        ea_num_iter,
        ea_topk_per_iter,
        ea_tree_size,
        draft_model_start
    );
}

void init_hier_eagle_w4a16_gm_rot_spec_w4a16_gm_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int draft_group_size,
    int min_draft_length,
    bool draft_cuda_graph,
    int ea_num_layers,
    int ea_num_iter,
    int ea_topk_per_iter,
    int ea_tree_size,
    bool draft_model_start,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new HierEagleW4A16GMRotSpecW4A16GMImpl<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        draft_vocab_size,
        draft_num_hidden_layers,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        draft_head_dim,
        draft_rms_norm_eps,
        draft_group_size,
        min_draft_length,
        draft_cuda_graph,
        ea_num_layers,
        ea_num_iter,
        ea_topk_per_iter,
        ea_tree_size,
        draft_model_start
    );
}


void init_hier_eagle_w4a16_gm_spec_w4a16_gm_dprefill_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int draft_group_size,
    int min_draft_length,
    bool draft_cuda_graph,
    int ea_num_layers,
    int ea_num_iter,
    int ea_topk_per_iter,
    int ea_tree_size,
    bool draft_model_start,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new HierEagleW4A16GMSpecW4A16GMDraftPrefillImpl<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        draft_vocab_size,
        draft_num_hidden_layers,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        draft_head_dim,
        draft_rms_norm_eps,
        draft_group_size,
        min_draft_length,
        draft_cuda_graph,
        ea_num_layers,
        ea_num_iter,
        ea_topk_per_iter,
        ea_tree_size,
        draft_model_start
    );
}

void init_hier_eagle_w4a16_gm_rot_spec_w4a16_gm_dprefill_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int draft_group_size,
    int min_draft_length,
    bool draft_cuda_graph,
    int ea_num_layers,
    int ea_num_iter,
    int ea_topk_per_iter,
    int ea_tree_size,
    bool draft_model_start,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new HierEagleW4A16GMRotSpecW4A16GMDraftPrefillImpl<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        draft_vocab_size,
        draft_num_hidden_layers,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        draft_head_dim,
        draft_rms_norm_eps,
        draft_group_size,
        min_draft_length,
        draft_cuda_graph,
        ea_num_layers,
        ea_num_iter,
        ea_topk_per_iter,
        ea_tree_size,
        draft_model_start
    );
}



void init_eagle3_model(
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    bool load_target_embed,
    int draft_vocab_size,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    DTYPE_SWITCH(torch_dtype, [&] {
        model = new Eagle3Impl<elem_type>(
            (ModelImpl<elem_type>*)model,
            draft_hidden_size,
            draft_intermediate_size,
            draft_num_attention_heads,
            draft_num_key_value_heads,
            load_target_embed,
            draft_vocab_size,
            num_iter,
            topk_per_iter,
            tree_size
        );
    });
}

void init_eagle3_w4a16_gptq_marlin_model(
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    bool load_target_embed,
    int draft_vocab_size,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    model = new Eagle3ImplBaseW4A16GPTQMarlin<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        load_target_embed,
        draft_vocab_size,
        num_iter,
        topk_per_iter,
        tree_size
    );
}


void init_eagle3_w4a16_gptq_marlin_dprefill_model(
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    bool load_target_embed,
    int draft_vocab_size,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    model = new Eagle3ImplBaseW4A16GPTQMarlinDraftPrefill<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        load_target_embed,
        draft_vocab_size,
        num_iter,
        topk_per_iter,
        tree_size
    );
}

void init_hier_eagle3_w4a16_gm_spec_w4a16_gm_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int draft_group_size,
    int min_draft_length,
    bool draft_cuda_graph,
    int ea_draft_hidden_size,
    int ea_draft_intermediate_size,
    int ea_draft_num_attention_heads,
    int ea_draft_num_key_value_heads,
    bool ea_load_target_embed,
    int ea_draft_vocab_size,
    int ea_num_iter,
    int ea_topk_per_iter,
    int ea_tree_size,
    bool draft_model_start,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new HierEagle3W4A16GMSpecW4A16GMImpl<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        draft_vocab_size,
        draft_num_hidden_layers,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        draft_head_dim,
        draft_rms_norm_eps,
        draft_group_size,
        min_draft_length,
        draft_cuda_graph,
        ea_draft_hidden_size,
        ea_draft_intermediate_size,
        ea_draft_num_attention_heads,
        ea_draft_num_key_value_heads,
        ea_load_target_embed,
        ea_draft_vocab_size,
        ea_num_iter,
        ea_topk_per_iter,
        ea_tree_size,
        draft_model_start
    );
}

void init_hier_eagle3_w4a16_gm_spec_w4a16_gm_dprefill_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int draft_group_size,
    int min_draft_length,
    bool draft_cuda_graph,
    int ea_draft_hidden_size,
    int ea_draft_intermediate_size,
    int ea_draft_num_attention_heads,
    int ea_draft_num_key_value_heads,
    bool ea_load_target_embed,
    int ea_draft_vocab_size,
    int ea_num_iter,
    int ea_topk_per_iter,
    int ea_tree_size,
    bool draft_model_start,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new HierEagle3W4A16GMSpecW4A16GMDraftPrefillImpl<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        draft_vocab_size,
        draft_num_hidden_layers,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        draft_head_dim,
        draft_rms_norm_eps,
        draft_group_size,
        min_draft_length,
        draft_cuda_graph,
        ea_draft_hidden_size,
        ea_draft_intermediate_size,
        ea_draft_num_attention_heads,
        ea_draft_num_key_value_heads,
        ea_load_target_embed,
        ea_draft_vocab_size,
        ea_num_iter,
        ea_topk_per_iter,
        ea_tree_size,
        draft_model_start
    );
}


int init_storage() {
    return model->init_storage();
}

void load_model(std::string name, std::uintptr_t param) {
    model->load_to_storage(name, reinterpret_cast<void*>(param));
}

void prefill(int input_length, int history_length, std::uintptr_t input, std::uintptr_t position_ids, std::uintptr_t output) {
    model->prefill(input_length, history_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), (void*)(output));
}

void decode(int input_length, int padded_length, std::uintptr_t input, std::uintptr_t position_ids, std::uintptr_t cache_length, std::uintptr_t mask_2d, std::uintptr_t output, bool cuda_graph) {
    if (cuda_graph) {
        if (graphCreated_padding_length != padded_length || graphCreated_input_length != input_length) {
            if (graphExec != nullptr) {
                cudaGraphExecDestroy(graphExec);
                graphExec = nullptr;
            }
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
                graph = nullptr;
            }
            cudaStreamBeginCapture(calc_stream.stream, cudaStreamCaptureModeGlobal);
            model->decode(input_length, padded_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(mask_2d), reinterpret_cast<void*>(output));
            cudaStreamEndCapture(calc_stream.stream, &graph);
            cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
            graphCreated_padding_length = padded_length;
            graphCreated_input_length = input_length;
        }
        cudaGraphLaunch(graphExec, calc_stream.stream);
    } else {
        model->decode(input_length, padded_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(mask_2d), reinterpret_cast<void*>(output));
    }
}

void draft_prefill(std::uintptr_t tree_draft_ids, std::uintptr_t tree_position_ids, std::uintptr_t cache_length) {
    model->draft_prefill(reinterpret_cast<int32_t*>(tree_draft_ids), reinterpret_cast<int32_t*>(tree_position_ids), reinterpret_cast<int32_t*>(cache_length));
}

void draft(std::uintptr_t tree_draft_ids, std::uintptr_t tree_position_ids, std::uintptr_t cache_length, std::uintptr_t attn_mask, std::uintptr_t tree_parent) {
    model->draft(reinterpret_cast<int32_t*>(tree_draft_ids), reinterpret_cast<int32_t*>(tree_position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(attn_mask), reinterpret_cast<int32_t*>(tree_parent));
}

int verify_and_fix(int num_tokens, std::uintptr_t pred, std::uintptr_t gt, std::uintptr_t position_ids, std::uintptr_t cache_length, std::uintptr_t attn_mask, std::uintptr_t tree_parent) {
    return model->verify(num_tokens, reinterpret_cast<int32_t*>(pred), reinterpret_cast<int32_t*>(gt), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(attn_mask), reinterpret_cast<int32_t*>(tree_parent));
}

void gptq_marlin_weight_repack(std::uintptr_t b_weight, std::uintptr_t perm, int64_t size_k, int64_t size_n, int64_t num_bits, bool has_perm, std::uintptr_t output) {
    return gptq_marlin_repack(
        reinterpret_cast<uint32_t const*>(b_weight),
        reinterpret_cast<uint32_t const*>(perm),
        size_k,
        size_n,
        num_bits,
        has_perm,
        reinterpret_cast<uint32_t*>(output),
        calc_stream.stream
    );
}

PYBIND11_MODULE(C, m) {
    // base bind
    m.def("init_base_model", &init_base_model, "Init base model");
    m.def("init_w8a8_base_model", &init_w8a8_base_model, "Init W8A8 base model");
    m.def("init_w4a16_gptq_marlin_base_model", &init_w4a16_gptq_marlin_base_model, "Init W4A16 base model");
    m.def("init_w4a8_qqq_base_model", &init_w4a8_qqq_base_model, "Init W4A8 per group base model");
    m.def("init_w4a8_qoq_chn_base_model", &init_w4a8_qoq_chn_base_model, "Init W4A8 per channel base model");
    m.def("init_w4a8_qoq_group_base_model", &init_w4a8_qoq_group_base_model, "Init W4A8 per group base model");

    // eagle bind
    m.def("init_eagle_model", &init_eagle_model, "Init eagle model");
    m.def("init_eagle_w8a8_model", &init_eagle_w8a8_model, "Init eagle W8A8 model");
    m.def("init_eagle_w4a16_gptq_marlin_model", &init_eagle_w4a16_gptq_marlin_model, "Init eagle W4A16 model");
    m.def("init_eagle_w4a16_gptq_marlin_rot_model", &init_eagle_w4a16_gptq_marlin_rot_model, "Init eagle W4A16 model");
    m.def("init_eagle_w4a8_qqq_rot_model", &init_eagle_w4a8_qqq_rot_model, "Init W4A8 per group base model");
    m.def("init_eagle_w4a8_qoq_chn_rot_model", &init_eagle_w4a8_qoq_chn_rot_model, "Init eagle W4A8 per channel model");
    m.def("init_eagle_w4a8_qoq_group_rot_model", &init_eagle_w4a8_qoq_group_rot_model, "Init W4A8 per group base model");

    m.def("init_eagle_w4a16_gptq_marlin_dprefill_model", &init_eagle_w4a16_gptq_marlin_dprefill_model, "Init eagle W4A16 model");
    m.def("init_eagle_w4a16_gptq_marlin_rot_dprefill_model", &init_eagle_w4a16_gptq_marlin_rot_dprefill_model, "Init eagle W4A16 model");

    // spec bind
    m.def("init_w4a16_gm_spec_w4a16_gm_model", &init_w4a16_gm_spec_w4a16_gm_model, "Init w4a16 spec v1 model");
    m.def("init_w4a16_gm_spec_w4a16_gm_dprefill_model", &init_w4a16_gm_spec_w4a16_gm_dprefill_model, "Init w4a16 spec v1 model");


    // hier spec bind
    m.def("init_hier_eagle_w4a16_gm_spec_w4a16_gm_model", &init_hier_eagle_w4a16_gm_spec_w4a16_gm_model, "init hier eagle gm spec gm model");
    m.def("init_hier_eagle_w4a16_gm_rot_spec_w4a16_gm_model", &init_hier_eagle_w4a16_gm_rot_spec_w4a16_gm_model, "init hier eagle rot gm spec gm model");
    m.def("init_hier_eagle_w4a16_gm_spec_w4a16_gm_dprefill_model", &init_hier_eagle_w4a16_gm_spec_w4a16_gm_dprefill_model, "init hier eagle gm spec gm model");
    m.def("init_hier_eagle_w4a16_gm_rot_spec_w4a16_gm_dprefill_model", &init_hier_eagle_w4a16_gm_rot_spec_w4a16_gm_dprefill_model, "init hier eagle rot gm spec gm model");
    
    // eagle3 
    m.def("init_eagle3_model", &init_eagle3_model, "Init eagle model");
    m.def("init_eagle3_w4a16_gptq_marlin_model", &init_eagle3_w4a16_gptq_marlin_model, "Init eagle model");

    m.def("init_eagle3_w4a16_gptq_marlin_dprefill_model", &init_eagle3_w4a16_gptq_marlin_dprefill_model, "Init eagle model");
    

    // hier eagle3
    m.def("init_hier_eagle3_w4a16_gm_spec_w4a16_gm_model", &init_hier_eagle3_w4a16_gm_spec_w4a16_gm_model, "Init casacde eagle3 model");
    m.def("init_hier_eagle3_w4a16_gm_spec_w4a16_gm_dprefill_model", &init_hier_eagle3_w4a16_gm_spec_w4a16_gm_dprefill_model, "Init casacde eagle3 model");
    
    
    m.def("draft_prefill", &draft_prefill, "draft prefill");
    m.def("init_storage", &init_storage, "Init storage");
    m.def("load_model", &load_model, "Load model");
    m.def("prefill", &prefill, "Prefill");
    m.def("decode", &decode, "Decode");
    m.def("draft", &draft, "Draft");
    m.def("verify_and_fix", &verify_and_fix, "Verify and fix");

    m.def("gptq_marlin_weight_repack", &gptq_marlin_weight_repack, "gptq marlin repack");
} 