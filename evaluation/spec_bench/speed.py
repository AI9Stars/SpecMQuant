import json
import argparse
from transformers import AutoTokenizer
import numpy as np


def speed(jsonl_file, jsonl_file_base, tokenizer, task=None, report=True):
    tokenizer=AutoTokenizer.from_pretrained(tokenizer)
    mt_bench_list = ["writing", "roleplay", "reasoning", "math" , "coding", "extraction", "stem", "humanities"]

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            if task=="overall":
                data.append(json_obj)
            elif task == "mt_bench":
                if json_obj["category"] in mt_bench_list:
                    data.append(json_obj)
            else:
                if json_obj["category"] == task:
                    data.append(json_obj)

    speeds=[]
    accept_lengths_list = []
    for datapoint in data:
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        accept_lengths_list.extend(datapoint["choices"][0]['accept_lengths'])
        speeds.append(tokens/times)


    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            if task=="overall":
                data.append(json_obj)
            elif task == "mt_bench":
                if json_obj["category"] in mt_bench_list:
                    data.append(json_obj)
            else:
                if json_obj["category"] == task:
                    data.append(json_obj)

    total_time=0
    total_token=0
    speeds0=[]
    for datapoint in data:
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds0.append(tokens / times)
        total_time+=times
        total_token+=tokens

    tokens_per_second = np.array(speeds).mean()
    tokens_per_second_baseline = np.array(speeds0).mean()
    speedup_ratio = np.array(speeds).mean()/np.array(speeds0).mean()

    if report:
        print("="*30, "Task: ", task, "="*30)
        print("#Mean accepted tokens: ", np.mean(accept_lengths_list))
        print('Tokens per second: ', tokens_per_second)
        print('Tokens per second for the baseline: ', tokens_per_second_baseline)
        print("Speedup ratio: ", speedup_ratio)
    return tokens_per_second, tokens_per_second_baseline, speedup_ratio, accept_lengths_list


def get_single_speedup(jsonl_file, jsonl_file_base, tokenizer_path):
    for subtask_name in ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag", "overall"]:
        speed(jsonl_file, jsonl_file_base, tokenizer_path, task=subtask_name)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file-path",
        type=str,
        help="The file path of evaluated Speculative Decoding methods.",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        help="The file path of evaluated baseline.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        help="The file path of evaluated baseline.",
    )
    parser.add_argument(
        "--mean-report",
        action="store_true",
        default=False,
        help="report mean speedup over different runs")

    args = parser.parse_args()

    get_single_speedup(jsonl_file=args.file_path, jsonl_file_base=args.base_path, tokenizer_path=args.tokenizer_path)