from .. import C
from .tree_drafter import LLM_with_tree_drafter

import torch
from transformers import PretrainedConfig

class Eagle3Config(PretrainedConfig):
    def __init__(
        self,
        **kwargs,
    ):
        self.draft_hidden_size          = kwargs.pop("hidden_size", None)
        self.draft_intermediate_size    = kwargs.pop("intermediate_size", None)
        self.draft_num_attention_heads  = kwargs.pop("num_attention_heads", None)
        self.draft_num_key_value_heads  = kwargs.pop("num_key_value_heads", None)
        self.draft_vocab_size           = kwargs.pop("draft_vocab_size", 1)

        # whether the config had a “target_hidden_size” key
        self.load_target_embed = not ("target_hidden_size" in kwargs)

        # let the base class handle everything else (e.g. target_hidden_size, hidden_act, etc.)
        super().__init__(**kwargs)

class LLM_with_eagle3(LLM_with_tree_drafter):
    def __init__(self,
                 eagle_path,
                 base_path,
                 num_iter=6,
                 topk_per_iter=10,
                 tree_size=60,
                 **kwargs):
        super().__init__(
            "eagle3", eagle_path, base_path,
            tree_size = tree_size,
            **kwargs
        )

        self.eagle_path = eagle_path
        self.eagle_config = Eagle3Config.from_pretrained(eagle_path)

        C.init_eagle3_model(
            self.eagle_config.draft_hidden_size,
            self.eagle_config.draft_intermediate_size,
            self.eagle_config.draft_num_attention_heads,
            self.eagle_config.draft_num_key_value_heads,
            self.eagle_config.load_target_embed,
            self.eagle_config.draft_vocab_size,
            num_iter,
            topk_per_iter,
            self.tree_size,
            self.dtype_int,
        )

    def _load(self, name, param, dtype=None, cls=None):
        if cls == self.drafter_type:
            if dtype is None:
                dtype = self.dtype
            
            if 'd2t' in name:
                param = param.contiguous().to(torch.int)
            elif 't2d' not in name:
                param = param.contiguous().to(dtype)

            
            if 'fc' in name:
                if 'weight' in name:
                    split_dim = param.shape[-1] // 3
                    param1 = param[..., :split_dim].contiguous()
                    param2 = param[..., split_dim: split_dim*2].contiguous()
                    param3 = param[..., split_dim*2:].contiguous()
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param1.data_ptr())
                    C.load_model(f"{cls}.{name.replace('fc', 'fc2')}", param2.data_ptr())
                    C.load_model(f"{cls}.{name.replace('fc', 'fc3')}", param3.data_ptr())
                else: # bias
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param.data_ptr())
            else:
                C.load_model(f"{cls}.{name}", param.data_ptr())
        else:
            super()._load(name, param, dtype)
