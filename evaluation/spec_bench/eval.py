# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import os
import time
import torch
import numpy as np
import shortuuid

from fastchat.llm_judge.common import load_questions
from tqdm import tqdm


def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        max_length,
        num_choices,
        teminators,
        **kwargs,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    # assert num_gpus_total % num_gpus_per_model == 0
    # use_ray = num_gpus_total // num_gpus_per_model > 1

    # if use_ray:
    #     import ray
    #     ray.init()
    #     get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
    #         get_model_answers
    #     ).remote
    # else:
    get_answers_func = get_model_answers


    # chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    # ans_handles = []
    # for i in range(0, len(questions), chunk_size):
        # ans_handles.append(
    get_answers_func(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        max_length,
        num_choices,
        teminators,
        **kwargs,
    )
        # )

    # if use_ray:
    #     ray.get(ans_handles)



@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        max_length,
        num_choices,
        teminators,
        **kwargs,
):
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]
    
    is_cascade = kwargs.pop('is_cascade', False)

    # warmup
    for wm_i in range(3):
        torch.manual_seed(0)
        messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        turns = []
        steps = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({
                "role": "user",
                "content": qs
            })
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer([prompt],add_special_tokens=False, return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids

            model_output = forward_func(
                inputs,
                model,
                tokenizer,
                max_new_tokens,
                max_length,
                teminators,
                **kwargs,
            )
            if is_cascade:
                if len(model_output) == 6:
                    output_ids, new_token, step, accept_length_tree, decode_time, cascade_accept_length_tree = model_output
                    draft_prefill_latency = None
                elif len(model_output) == 7:
                    output_ids, new_token, step, accept_length_tree, decode_time, cascade_accept_length_tree, draft_prefill_latency = model_output
                else:
                    raise ValueError("Invalid model output")
            else:
                if len(model_output) == 5:
                    output_ids, new_token, step, accept_length_tree, decode_time = model_output
                    draft_prefill_latency = None
                elif len(model_output) == 6:
                    output_ids, new_token, step, accept_length_tree, decode_time, draft_prefill_latency = model_output
                else:
                    raise ValueError("Invalid model output")

            if len(teminators)>0:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in teminators
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()
            

            # except RuntimeError as e:
            #     print("ERROR question ID: ", question["question_id"])
            #     output = "ERROR"

            turns.append(output)
            steps.append(int(step))
            new_tokens.append(int(new_token))
            wall_time.append(decode_time)
            messages.append({
                "role": "assistant",
                "content": output
            })
        
        print(f"warmup {wm_i} done")

            
    print('Warmup done')

    accept_lengths_tree = []
    cascade_accept_lengths_tree = []
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            cur_accept_lengths_tree = []
            cur_cascade_accept_lengths_tree = []
            torch.manual_seed(i)
            messages = [
                {"role": "system",
                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            draft_prefill_latency_list = []
            generate_speed = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({
                    "role": "user",
                    "content": qs
                })
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer([prompt],add_special_tokens=False, return_tensors="pt").to("cuda")
                input_ids = inputs.input_ids
                # try:
                # torch.cuda.synchronize()
                # start_time = time.time()
                model_output = forward_func(
                    inputs,
                    model,
                    tokenizer,
                    max_new_tokens,
                    max_length,
                    teminators,
                    **kwargs,
                )
                if is_cascade:
                    if len(model_output) == 6:
                        output_ids, new_token, step, accept_length_tree, decode_time, cascade_accept_length_tree = model_output
                        draft_prefill_latency = None
                    elif len(model_output) == 7:
                        output_ids, new_token, step, accept_length_tree, decode_time, cascade_accept_length_tree, draft_prefill_latency = model_output
                    else:
                        raise ValueError("Invalid model output")
                    cascade_accept_lengths_tree.extend(cascade_accept_length_tree)
                    cur_cascade_accept_lengths_tree.extend(cascade_accept_length_tree)
                else:
                    if len(model_output) == 5:
                        output_ids, new_token, step, accept_length_tree, decode_time = model_output
                        draft_prefill_latency = None
                    elif len(model_output) == 6:
                        output_ids, new_token, step, accept_length_tree, decode_time, draft_prefill_latency = model_output
                    else:
                        raise ValueError("Invalid model output")
                # torch.cuda.synchronize()
                # total_time = time.time() - start_time
                accept_lengths_tree.extend(accept_length_tree)

                if len(teminators)>0:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in teminators
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                # except RuntimeError as e:
                #     print("ERROR question ID: ", question["question_id"])
                #     output = "ERROR"

                turns.append(output)
                steps.append(int(step))
                new_tokens.append(int(new_token))
                wall_time.append(decode_time)
                if draft_prefill_latency is not None:
                    draft_prefill_latency_list.append(draft_prefill_latency)
                generate_speed.append(int(new_token) / decode_time)
                cur_accept_lengths_tree.extend(accept_length_tree)
                messages.append({
                    "role": "assistant",
                    "content": output
                })
            # torch.cuda.empty_cache()
            if is_cascade:
                choice_ans = {"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                                "accept_lengths": cur_accept_lengths_tree, "cascade_accept_lengths": cur_cascade_accept_lengths_tree, "generate_speed": generate_speed}
            else:
                choice_ans = {"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                                "accept_lengths": cur_accept_lengths_tree, "generate_speed": generate_speed}
            if len(draft_prefill_latency_list) > 0:
                choice_ans["draft_prefill_latency"] = draft_prefill_latency_list
            
            choices.append(choice_ans)

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))
    if is_cascade:
        print("#Mean cascade accepted tokens: ", np.mean(cascade_accept_lengths_tree))


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

