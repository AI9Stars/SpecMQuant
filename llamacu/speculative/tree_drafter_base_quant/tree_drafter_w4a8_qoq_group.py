from ... import C
from ...llama_w4a8_qoq_group import W4A8QoQGroupLLM

import torch
from ..tree_drafter import *
import time



class W4A8QoQGroupLLM_with_tree_drafter(W4A8QoQGroupLLM):
    def __init__(self,
                 drafter_type, drafter_path, base_path,
                 tree_size,
                 **kwargs):
        super().__init__(base_path, **kwargs)

        self.drafter_type = drafter_type
        self.drafter_path = drafter_path
        self.base_path = base_path

        self.tree_size = tree_size
        self.tree_draft_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_position_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_gt_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_attn_mask = torch.empty((tree_size), dtype=torch.uint64, device="cuda")
        self.tree_parent = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_position_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")

        self.cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")

    def load_from_hf(self):
        self._load_from_ckpt(self.drafter_path, cls=self.drafter_type)
        super().load_from_hf()

    def generate(self, input_ids, generation_length=100, teminators=[]):
        assert input_ids.dtype == torch.int32

        prefix_length = input_ids.numel()
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        logits = self.prefill(input_ids, position_ids)
        self.tree_draft_ids[:1].copy_(logits[0].argmax(dim=-1))

        tokens = torch.empty((generation_length), dtype=torch.int32, device="cuda")
        tokens[0].copy_(self.tree_draft_ids[0])
        accept_lengths = []
        i = 0
        model_step = 0
        terminal = False
        torch.cuda.synchronize()
        start_time = time.time()
        while i < generation_length-1 and not terminal:
            self.cache_length[0] = prefix_length + i

            # torch.cuda.nvtx.range_push(f"draft")
            C.draft(self.tree_draft_ids.data_ptr(), self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(), self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr())
            # torch.cuda.nvtx.range_pop()

            logits = self.decode(self.tree_draft_ids, self.tree_position_ids, self.cache_length, mask_2d=self.tree_attn_mask)
            self.tree_gt_ids.copy_(logits.argmax(dim=-1))

            # torch.cuda.nvtx.range_push(f"verify")
            accept_length = C.verify_and_fix(
                self.tree_draft_ids.numel(), self.tree_draft_ids.data_ptr(), self.tree_gt_ids.data_ptr(),
                self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(),
                self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr()
            )
            # torch.cuda.nvtx.range_pop()

            model_step += 1
            accept_lengths.append(accept_length)
            for temin in teminators:
                if temin in self.tree_draft_ids[:accept_length]:
                    terminal = True
            append_length = min(accept_length, generation_length - 1 - i)
            tokens[1+i:1+i+append_length].copy_(self.tree_draft_ids[:append_length])
            self.tree_draft_ids[0] = self.tree_draft_ids[accept_length - 1]
            i += accept_length
        torch.cuda.synchronize()
        decode_time = time.time() - start_time
        tokens = tokens[:1+i].tolist()
        return tokens, accept_lengths, model_step, decode_time