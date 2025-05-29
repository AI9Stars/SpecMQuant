from ... import C
from ..eagle import EagleConfig
from ..tree_drafter_base_quant.tree_drafter_w4a16_gptq_marlin import W4A16GPTQMarlinLLM_with_tree_drafter

import torch


class W4A16GPTQMarlinLLM_with_eagle(W4A16GPTQMarlinLLM_with_tree_drafter):
    def __init__(self,
                 eagle_path,
                 base_path,
                 num_iter=6,
                 topk_per_iter=10,
                 tree_size=60,
                 draft_prefill_sep=False,
                 rotation=False,
                 **kwargs):
        super().__init__(
            "eagle", eagle_path, base_path,
            tree_size = tree_size,
            draft_prefill_sep = draft_prefill_sep,
            **kwargs
        )

        self.eagle_path = eagle_path
        self.eagle_config = EagleConfig.from_pretrained(eagle_path)
        self.rotation = rotation

        if self.rotation:
            if self.draft_prefill_sep:
                C.init_eagle_w4a16_gptq_marlin_rot_dprefill_model(
                    self.eagle_config.eagle_num_layers,
                    num_iter,
                    topk_per_iter,
                    self.tree_size,
                    self.dtype_int,
                )
            else:
                C.init_eagle_w4a16_gptq_marlin_rot_model(
                    self.eagle_config.eagle_num_layers,
                    num_iter,
                    topk_per_iter,
                    self.tree_size,
                    self.dtype_int,
                )
        else:
            if self.draft_prefill_sep:
                C.init_eagle_w4a16_gptq_marlin_dprefill_model(
                    self.eagle_config.eagle_num_layers,
                    num_iter,
                    topk_per_iter,
                    self.tree_size,
                    self.dtype_int,
                )
            else:
                C.init_eagle_w4a16_gptq_marlin_model(
                    self.eagle_config.eagle_num_layers,
                    num_iter,
                    topk_per_iter,
                    self.tree_size,
                    self.dtype_int,
                )

    def _load(self, name, param, dtype=None, cls=None):
        if cls == self.drafter_type:
            if dtype is None:
                dtype = self.dtype
            param = param.contiguous().to(dtype)
            if (not self.rotation) and 'embed_tokens' in name:
                return
            if 'fc' in name:
                if 'weight' in name:
                    param1 = param[..., :param.shape[-1] // 2].contiguous()
                    param2 = param[..., param.shape[-1] // 2:].contiguous()
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param1.data_ptr())
                    C.load_model(f"{cls}.{name.replace('fc', 'fc2')}", param2.data_ptr())
                else: # bias
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param.data_ptr())
            else:
                C.load_model(f"{cls}.{name}", param.data_ptr())
        else:
            super()._load(name, param, dtype)
