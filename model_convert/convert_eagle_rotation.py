import os
import re
import shutil
import torch
import json
from safetensors.torch import load_file,save_file
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--eagle-path", type=str, required=True, help="Path to the EAGLE model")
parser.add_argument("--base-model-path", type=str, required=True, help="Path to the LLaMA3 model")
parser.add_argument("--rotation-path", type=str, required=True, help="Path to the rotation weights")
parser.add_argument("--output-path", type=str, required=True, help="Path to save the converted model")

def convert(eagle_path, base_model_path, rotation_path, output_path):
    eagle_file = os.path.join(eagle_path, "pytorch_model.bin")
    eagle_ckpt = torch.load(eagle_file)
    
    index_file = os.path.join(base_model_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        weight_map = json.load(f)["weight_map"]
        lm_head_file = weight_map["lm_head.weight"]
        rms_norm_file = weight_map["model.norm.weight"]

    if lm_head_file == rms_norm_file:
        lm_head_model_ckpt = load_file(os.path.join(base_model_path, lm_head_file))
        rms_norm_model_ckpt = lm_head_model_ckpt
    else:
        lm_head_model_ckpt = load_file(os.path.join(base_model_path, lm_head_file))
        rms_norm_model_ckpt = load_file(os.path.join(base_model_path, rms_norm_file))


    lm_head_weight = lm_head_model_ckpt["lm_head.weight"]
    
    new_eagle_ckpt = {}

    def transform_rms_norm_and_rotation(norm_weight, rotation_weight):
        """Fuse the weight multiplication of rms norm into the next adjacent linear modules.

        Args:
            norm (`nn.LayerNorm` or `RMSNorm`):
                normalization module.
            next_modules (`Iterable[nn.Linear]`):
                modules after the normalization module.
        """
        ln_w = norm_weight.to(dtype=torch.float64)

        dtype = rotation_weight.dtype
        fc_w = rotation_weight.to(dtype=torch.float64)
        ln_w = ln_w.to(fc_w.device)
        rotation_weight_norm = (fc_w * ln_w.unsqueeze(1)).to(dtype=dtype)
        return rotation_weight_norm


    rotation_weights = torch.load(rotation_path)
    rms_norm_weights = rms_norm_model_ckpt["model.norm.weight"]


    rotation_weights_norm = transform_rms_norm_and_rotation(rms_norm_weights, rotation_weights.clone())
    new_eagle_ckpt['rms_norm_rotation.weight'] = rotation_weights_norm


    for key in eagle_ckpt:
        new_eagle_ckpt[key] = eagle_ckpt[key].clone().detach()
    
    new_eagle_ckpt['lm_head.weight'] = lm_head_weight.clone().detach()

    torch.save(new_eagle_ckpt, os.path.join(output_path, "pytorch_model.bin"))

    src_path = os.path.join(eagle_path, "config.json")
    dst_path = os.path.join(output_path, "config.json")
    shutil.copy2(src_path, dst_path)
    


if __name__=="__main__":

    args = parser.parse_args()
    eagle_path = args.eagle_path
    base_model_path = args.base_model_path
    rotation_path = args.rotation_path
    output_path = args.output_path
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    convert(eagle_path, base_model_path, rotation_path, output_path)
    
    