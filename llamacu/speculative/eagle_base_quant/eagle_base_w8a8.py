from ... import C
from ..eagle import EagleConfig
from ..tree_drafter_base_quant.tree_drafter_w8a8 import W8A8LLM_with_tree_drafter

import torch


class W8A8LLM_with_eagle(W8A8LLM_with_tree_drafter):
    def __init__(self,
                 eagle_path,
                 base_path,
                 num_iter=6,
                 topk_per_iter=10,
                 tree_size=60,
                 **kwargs):
        super().__init__(
            "eagle", eagle_path, base_path,
            tree_size = tree_size,
            **kwargs
        )

        self.eagle_path = eagle_path
        self.eagle_config = EagleConfig.from_pretrained(eagle_path)

        C.init_eagle_w8a8_model(
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
            if 'embed_tokens' in name:
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
