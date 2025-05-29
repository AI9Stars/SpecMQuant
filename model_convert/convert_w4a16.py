import torch
from safetensors.torch import load_file, save_file
import os, glob, shutil
import re
from typing import List
import argparse
from transformers import AutoConfig
from llamacu import C

parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, required=True, help="Path to the original model")
parser.add_argument("--quant-path", type=str, required=True, help="Path to the AutoGPTQ model")
parser.add_argument("--output-path", type=str, required=True, help="Path to save the converted model")

TILESIZE = 16

def convert_w4a16_checkpoint(orig_model_path, quant_path, output_path):

    config = AutoConfig.from_pretrained(quant_path)
    group_size = config.quantization_config['group_size']
    
    model_path = glob.glob(os.path.join(quant_path, "*.safetensors"))[0]

    autogptq_weigths = load_file(model_path)

    gptq_convert_dict = {
        "model.layers.{}.self_attn.q_proj.qweight": ["model.layers.{}.self_attn.q_proj.scales", "model.layers.{}.self_attn.q_proj.g_idx", "model.layers.{}.self_attn.q_proj.qzeros"], 
        "model.layers.{}.self_attn.k_proj.qweight":["model.layers.{}.self_attn.k_proj.scales", "model.layers.{}.self_attn.k_proj.g_idx", "model.layers.{}.self_attn.k_proj.qzeros"],
        "model.layers.{}.self_attn.v_proj.qweight":["model.layers.{}.self_attn.v_proj.scales", "model.layers.{}.self_attn.v_proj.g_idx", "model.layers.{}.self_attn.v_proj.qzeros"],
        "model.layers.{}.self_attn.o_proj.qweight":["model.layers.{}.self_attn.o_proj.scales", "model.layers.{}.self_attn.o_proj.g_idx", "model.layers.{}.self_attn.o_proj.qzeros"],
        "model.layers.{}.mlp.gate_proj.qweight":["model.layers.{}.mlp.gate_proj.scales", "model.layers.{}.mlp.gate_proj.g_idx", "model.layers.{}.mlp.gate_proj.qzeros"],
        "model.layers.{}.mlp.up_proj.qweight": ["model.layers.{}.mlp.up_proj.scales", "model.layers.{}.mlp.up_proj.g_idx", "model.layers.{}.mlp.up_proj.qzeros"],
        "model.layers.{}.mlp.down_proj.qweight": ["model.layers.{}.mlp.down_proj.scales", "model.layers.{}.mlp.down_proj.g_idx", "model.layers.{}.mlp.down_proj.qzeros"],
    }

    convert_checkpoint = {}
    processed_keys = set()

    def get_scale_perms():
        scale_perm: List[int] = []
        for i in range(8):
            scale_perm.extend([i + 8 * j for j in range(8)])
        scale_perm_single: List[int] = []
        for i in range(4):
            scale_perm_single.extend(
                [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
        return scale_perm, scale_perm_single


    def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                            group_size: int) -> torch.Tensor:

        scale_perm, scale_perm_single = get_scale_perms()
        if group_size < size_k and group_size != -1:
            s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
        else:
            s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
        s = s.reshape((-1, size_n)).contiguous()

        return s

    for gptq_key in autogptq_weigths:
        if gptq_key in processed_keys:
            continue
        elif "layers" in gptq_key:
            abstract_key = re.sub(r'(\d+)', '{}', gptq_key)
            layer_num = re.search(r'\d+', gptq_key).group(0)
            if "q_proj" in abstract_key:
                if abstract_key.endswith('qweight'):
                    k_key = gptq_key.replace('q_proj', 'k_proj')
                    v_key = gptq_key.replace('q_proj', 'v_proj')
                    
                    q_weight = autogptq_weigths[gptq_key].clone().cuda()
                    k_weight = autogptq_weigths[k_key].clone().cuda()
                    v_weight = autogptq_weigths[v_key].clone().cuda()
                    x = torch.cat([q_weight, k_weight, v_weight], dim=-1)
                    shape_0 = x.shape[0]*8
                    shape_1 = x.shape[1]

                    packed_data = torch.zeros((shape_0//TILESIZE, shape_1*TILESIZE//8), dtype=torch.int32, device=x.device)
                    C.gptq_marlin_weight_repack(
                        x.data_ptr(),
                        torch.Tensor([]).to( device="cuda", dtype=torch.int32).data_ptr(),
                        shape_0,
                        shape_1,
                        4,
                        False,
                        packed_data.data_ptr(),
                    )
                
                    convert_checkpoint[gptq_key.replace("q_proj", "qkv_proj")] = packed_data.cpu()

                    processed_keys.add(gptq_key)
                    processed_keys.add(k_key)
                    processed_keys.add(v_key)

                    for q_keys in gptq_convert_dict[abstract_key]:
                        if q_keys.endswith("scales"):
                            k_q_keys = q_keys.replace("q_proj", "k_proj")
                            v_q_keys = q_keys.replace("q_proj", "v_proj")   

                            scales_x_q = autogptq_weigths[q_keys.format(layer_num)].clone().cuda()
                            scales_x_k = autogptq_weigths[k_q_keys.format(layer_num)].clone().cuda()
                            scales_x_v = autogptq_weigths[v_q_keys.format(layer_num)].clone().cuda()
                            scales_x = torch.cat([scales_x_q, scales_x_k, scales_x_v], dim=-1)
                            scales_x.data = marlin_permute_scales(scales_x.data.contiguous(),
                                    size_k=shape_0,
                                    size_n=shape_1,
                                    group_size=group_size)
                            convert_checkpoint[q_keys.format(layer_num).replace("q_proj", "qkv_proj")] = scales_x.cpu()
                        
                        processed_keys.add(q_keys.format(layer_num))
                        processed_keys.add(q_keys.replace("q_proj", "k_proj").format(layer_num))
                        processed_keys.add(q_keys.replace("q_proj", "v_proj").format(layer_num))
            elif "gate_proj" in abstract_key:
                if abstract_key.endswith('qweight'):
                    up_key = gptq_key.replace('gate_proj', 'up_proj')
                    
                    gate_weight = autogptq_weigths[gptq_key].clone().cuda()
                    up_weight = autogptq_weigths[up_key].clone().cuda()

                    x = torch.cat([gate_weight, up_weight], dim=-1)
                    shape_0 = x.shape[0]*8
                    shape_1 = x.shape[1]
                    packed_data = torch.zeros((shape_0//TILESIZE, shape_1*TILESIZE//8), dtype=torch.int32, device=x.device)
                    C.gptq_marlin_weight_repack(
                        x.data_ptr(),
                        torch.Tensor([]).to( device="cuda", dtype=torch.int32).data_ptr(),
                        shape_0,
                        shape_1,
                        4,
                        False,
                        packed_data.data_ptr(),
                    )
                
                    convert_checkpoint[gptq_key.replace("gate_proj", "gate_up_proj")] = packed_data.cpu()

                    processed_keys.add(gptq_key)
                    processed_keys.add(up_key)

                    for q_keys in gptq_convert_dict[abstract_key]:
                        if q_keys.endswith("scales"):
                            up_q_keys = q_keys.replace("gate_proj", "up_proj")

                            scales_x_gate = autogptq_weigths[q_keys.format(layer_num)].clone().cuda()
                            scales_x_up = autogptq_weigths[up_q_keys.format(layer_num)].clone().cuda()
                            scales_x = torch.cat([scales_x_gate, scales_x_up], dim=-1)
                            scales_x.data = marlin_permute_scales(scales_x.data.contiguous(),
                                    size_k=shape_0,
                                    size_n=shape_1,
                                    group_size=group_size)
                            convert_checkpoint[q_keys.format(layer_num).replace("gate_proj", "gate_up_proj")] = scales_x.cpu()
                        
                        processed_keys.add(q_keys.format(layer_num))
                        processed_keys.add(q_keys.replace("gate_proj", "up_proj").format(layer_num))

            elif "down_proj" in abstract_key or "o_proj" in abstract_key:
                if abstract_key.endswith('qweight'):
                    x = autogptq_weigths[gptq_key].clone().cuda()

                    shape_0 = x.shape[0]*8
                    shape_1 = x.shape[1]

                    packed_data = torch.zeros((shape_0//TILESIZE, shape_1*TILESIZE//8), dtype=torch.int32, device=x.device)
                    C.gptq_marlin_weight_repack(
                        x.data_ptr(),
                        torch.Tensor([]).to( device="cuda", dtype=torch.int32).data_ptr(),
                        shape_0,
                        shape_1,
                        4,
                        False,
                        packed_data.data_ptr(),
                    )
                
                    convert_checkpoint[gptq_key] = packed_data.cpu()

                    processed_keys.add(gptq_key)

                    for q_keys in gptq_convert_dict[abstract_key]:
                        if q_keys.endswith("scales"):

                            scales_x = autogptq_weigths[q_keys.format(layer_num)].clone().cuda()
                            scales_x.data = marlin_permute_scales(scales_x.data.contiguous(),
                                    size_k=shape_0,
                                    size_n=shape_1,
                                    group_size=group_size)
                            convert_checkpoint[q_keys.format(layer_num)] = scales_x.cpu()

                        processed_keys.add(q_keys.format(layer_num))

            elif "post_attention_layernorm" in gptq_key or "input_layernorm" in gptq_key:
                convert_checkpoint[gptq_key] = autogptq_weigths[gptq_key].clone()
        else:  
            convert_checkpoint[gptq_key] = autogptq_weigths[gptq_key].clone()

    save_file(convert_checkpoint, os.path.join(output_path, f"model_gptq.safetensors"))
    # copy quantization config
    config_list = glob.glob(os.path.join(quant_path, "*config.json"))
    for config_file in config_list:
        # copy config to output path
        config_filename = os.path.basename(config_file)
        dst_path = os.path.join(output_path, config_filename)
        shutil.copy2(config_file, dst_path)
    
    # copy tokenizer
    tokenizer_list = glob.glob(os.path.join(orig_model_path, "tokenizer*"))
    for tokenizer_file in tokenizer_list:
        # copy config to output path
        tokenizer_filename = os.path.basename(tokenizer_file)
        dst_path = os.path.join(output_path, tokenizer_filename)
        shutil.copy2(tokenizer_file, dst_path)
    
    # copy "special_tokens_map.json"
    special_tokens_map_file = glob.glob(os.path.join(orig_model_path, "special_tokens_map.json"))[0]
    special_tokens_map_basename = os.path.basename(special_tokens_map_file)
    dst_path = os.path.join(output_path, special_tokens_map_basename)
    shutil.copy2(special_tokens_map_file, dst_path)
    
if __name__=="__main__":
    
    args = parser.parse_args()
    orig_model_path = args.model_path
    quant_path = args.quant_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    convert_w4a16_checkpoint(orig_model_path, quant_path, output_path)