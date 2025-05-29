import torch
from safetensors.torch import load_file, save_file
import re
from typing import List
import json
import os, shutil
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--quant-path", type=str, required=True, help="Path to the Quant model")
parser.add_argument("--output-path", type=str, required=True, help="Path to save the converted model")

def convert_w4a8_qqq(quant_path, output_path):
    index_file = os.path.join(quant_path, "model.safetensors.index.json")

    with open(index_file, "r") as f:
        file_list = set(json.load(f)["weight_map"].values())
        file_list = [os.path.join(quant_path, file) for file in file_list]

    quant_dict = {
        "model.layers.{}.self_attn.q_proj.B": ["model.layers.{}.self_attn.q_proj.s_channel"],
        "model.layers.{}.mlp.gate_proj.B": ["model.layers.{}.mlp.gate_proj.s_channel"],
    }


    input_ckpt = {}
    for file in file_list:
        print(f"load from {file}")
        ckpt = load_file(file)
        input_ckpt.update(ckpt)

    output_ckpt = {}
    processed_keys = set()
    for key in input_ckpt.keys():
        if key in processed_keys:
            continue
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key)
            layer_num = re.search(r'\d+', key).group(0)

            if "q_proj.B" in abstract_key:
                k_key = key.replace('q_proj', 'k_proj')
                v_key = key.replace('q_proj', 'v_proj')
                
                
                q_weight = input_ckpt[key]
                k_weight = input_ckpt[k_key]
                v_weight = input_ckpt[v_key]


                x = torch.cat([q_weight, k_weight, v_weight], dim=-1)
                output_ckpt[key.replace('q_proj', 'qkv_proj')] = x

                if k_key in output_ckpt:
                    del output_ckpt[k_key]
                if v_key in output_ckpt:
                    del output_ckpt[v_key]

                processed_keys.add(key)
                processed_keys.add(k_key)
                processed_keys.add(v_key)

                for quant_key in quant_dict[abstract_key]:
                    # quant_key = quant_key.format(layer_num)
                    if "s_channel" in quant_key:
                        k_quant_key = quant_key.replace('q_proj', 'k_proj')
                        v_quant_key = quant_key.replace('q_proj', 'v_proj')
                        
                        scale_q = input_ckpt[quant_key.format(layer_num)]
                        scale_k = input_ckpt[k_quant_key.format(layer_num)]
                        scale_v = input_ckpt[v_quant_key.format(layer_num)]

                        scale_x = torch.cat([scale_q, scale_k, scale_v], dim=-1)

                        output_ckpt[quant_key.format(layer_num).replace('q_proj', 'qkv_proj')] = scale_x

                        if quant_key.format(layer_num) in output_ckpt:
                            del output_ckpt[quant_key.format(layer_num)]
                        if k_quant_key.format(layer_num) in output_ckpt:
                            del output_ckpt[k_quant_key.format(layer_num)]
                        if v_quant_key.format(layer_num) in output_ckpt:
                            del output_ckpt[v_quant_key.format(layer_num)]

                        processed_keys.add(quant_key.format(layer_num))
                        processed_keys.add(k_quant_key.format(layer_num))
                        processed_keys.add(v_quant_key.format(layer_num))
            elif "gate_proj.B" in abstract_key:
                up_key = key.replace('gate_proj', 'up_proj')
                
                gate_weight = input_ckpt[key]
                up_weight = input_ckpt[up_key]


                x = torch.cat([gate_weight, up_weight], dim=-1)
                output_ckpt[key.replace('gate_proj', 'gate_up_proj')] = x

                if up_key in output_ckpt:
                    del output_ckpt[up_key]

                processed_keys.add(key)
                processed_keys.add(up_key)

                for quant_key in quant_dict[abstract_key]:
                    # quant_key = quant_key.format(layer_num)
                    if "s_channel" in quant_key:
                        up_quant_key = quant_key.replace('gate_proj', 'up_proj')
                        
                        scale_gate = input_ckpt[quant_key.format(layer_num)]
                        scale_up = input_ckpt[up_quant_key.format(layer_num)]

                        scale_x = torch.cat([scale_gate, scale_up], dim=-1)

                        output_ckpt[quant_key.format(layer_num).replace('gate_proj', 'gate_up_proj')] = scale_x

                        if quant_key.format(layer_num) in output_ckpt:
                            del output_ckpt[quant_key.format(layer_num)]
                        if up_quant_key.format(layer_num) in output_ckpt:
                            del output_ckpt[up_quant_key.format(layer_num)]
                        

                        processed_keys.add(quant_key.format(layer_num))
                        processed_keys.add(up_quant_key.format(layer_num))
            else:
                output_ckpt[key] = input_ckpt[key].clone()
        else:
            output_ckpt[key] = input_ckpt[key].clone()

    save_file(output_ckpt, os.path.join(output_path, "model.safetensors")) 

    for name in os.listdir(quant_path):
        src_path = os.path.join(quant_path, name)

        if os.path.isfile(src_path) and 'safetensors' not in name:
            dst_path = os.path.join(output_path, name)
            shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    args = parser.parse_args()
    quant_path = args.quant_path
    output_path = args.output_path
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    convert_w4a8_qqq(quant_path, output_path)
