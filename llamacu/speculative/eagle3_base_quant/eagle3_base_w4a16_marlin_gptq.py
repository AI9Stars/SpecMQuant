from ... import C
from ..eagle3 import Eagle3Config
from ..tree_drafter_base_quant.tree_drafter_w4a16_gptq_marlin import W4A16GPTQMarlinLLM_with_tree_drafter

import torch


class W4A16GPTQMarlinLLM_with_eagle3(W4A16GPTQMarlinLLM_with_tree_drafter):
    def __init__(self,
                 eagle_path,
                 base_path,
                 num_iter=6,
                 topk_per_iter=10,
                 tree_size=60,
                 draft_prefill_sep=False,
                 **kwargs):
        super().__init__(
            "eagle3", eagle_path, base_path,
            tree_size = tree_size,
            draft_prefill_sep=draft_prefill_sep,
            **kwargs
        )

        self.eagle_path = eagle_path
        self.eagle_config = Eagle3Config.from_pretrained(eagle_path)

        if draft_prefill_sep:
            C.init_eagle3_w4a16_gptq_marlin_dprefill_model(
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
        else:
            C.init_eagle3_w4a16_gptq_marlin_model(
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
