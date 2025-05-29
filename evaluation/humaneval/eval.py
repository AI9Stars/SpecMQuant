import os
import json
from tqdm import tqdm

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

from transformers import AutoTokenizer

import time, torch

import argparse, re
import numpy as np
import shortuuid

from .sanitize import sanitize


def entry_point(
    problem_file: str,
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(
        sample_file, k, n_workers, timeout, problem_file
    )

    return results



def count_indent(text: str) -> int:
    count = 0
    for char in text:
        if char == " ":
            count += 1
        else:
            break
    return count


def fix_indents(text: str, multiple: int = 2):
    outputs = []
    for line in text.split("\n"):
        while count_indent(line) % multiple != 0:
            line = " " + line
        outputs.append(line)
    return "\n".join(outputs)


def test_fix_indents():
    text = "   # TODO: Implement separate_paren_groups\nreturn []"
    print(fix_indents(text))


######

# borrow from evalplus

EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]
EOS += ["\n```\n"]


# Model instructions
instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def make_raw_chat_prompt(
    task_prompt: str,
    instruction_prefix: str,
    response_prefix: str,
    tokenizer,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer.chat_template is None:
        return task_prompt

    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"

    task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""
    task_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]

    return task_prompt

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
    questions = read_problems(question_file)

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
        question_file,
        **kwargs,
    )


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
    question_file,
    **kwargs,
):

    is_cascade = kwargs.pop('is_cascade', False)

    entry = questions['HumanEval/0']
    warmup_times = 3
    for wm_i in range(warmup_times):
        prompt = entry["prompt"]
        
        task_prompt = make_raw_chat_prompt(
            prompt,
            instruction_prefix,
            response_prefix,
            tokenizer,
        )

        input_ids = tokenizer.encode(task_prompt, return_tensors='pt').to("cuda").view(1, -1)
        output_ids, new_token, step, accept_length_tree, decode_time, *rest = forward_func(
            input_ids, 
            model, 
            tokenizer, 
            max_new_tokens, 
            max_length,
            teminators
        )
        print(f"warmup {wm_i} done")
    print("Warmup done")


    eval_samples = []
    accept_lengths_tree = []
    cascade_accept_lengths_tree = []
    total_new_tokens = 0
    total_time = 0
    progress_bar = tqdm(total=len(questions) * num_choices, desc="Generating samples")
    for task_id in questions:
        choices = []
        for sample_id in range(num_choices):
            torch.manual_seed(sample_id)
            cur_accept_lengths_tree = []
            cur_cascade_accept_lengths_tree = []
            prompt = questions[task_id]["prompt"]
            
            task_prompt = make_raw_chat_prompt(
                prompt,
                instruction_prefix,
                response_prefix,
                tokenizer,
            )
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            generate_speed = []
            
            input_ids = tokenizer.encode(task_prompt, return_tensors='pt').to("cuda").view(1, -1)
            # torch.cuda.synchronize()
            # start_time = time.time()
            if is_cascade:
                output_ids, new_token, step, accept_length_tree, decode_time, cascade_accept_length_tree, *rest = forward_func(
                    input_ids,
                    model,
                    tokenizer,
                    max_new_tokens,
                    max_length,
                    teminators,
                    **kwargs,
                )
                cascade_accept_lengths_tree.extend(cascade_accept_length_tree)
                cur_cascade_accept_lengths_tree.extend(cascade_accept_length_tree)
            else:
                output_ids, new_token, step, accept_length_tree, decode_time, *rest = forward_func(
                    input_ids,
                    model,
                    tokenizer,
                    max_new_tokens,
                    max_length,
                    teminators,
                    **kwargs,
                )
            # torch.cuda.synchronize()
            # cur_time = time.time() - start_time
            accept_lengths_tree.extend(accept_length_tree)
            
            output_completion = tokenizer.decode(output_ids, skip_special_tokens=True)
            min_index = 10000
            for eos in EOS:
                if eos in output_completion:
                    min_index = min(min_index, output_completion.index(eos))
            completion = output_completion[:min_index].replace("\t", "    ")
            # fix_completion = fix_indents(completion)
            sanitized_completion = sanitize(completion, questions[task_id]["entry_point"])

            eval_sample = dict(task_id=task_id, completion=sanitized_completion) 
            eval_samples.append(eval_sample)
            
            turns.append(output_completion)
            steps.append(int(step))
            new_tokens.append(int(new_token))
            wall_time.append(decode_time)
            generate_speed.append(int(new_token) / decode_time)
            cur_accept_lengths_tree.extend(accept_length_tree)

            if is_cascade:
                choices.append({"index": sample_id, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                                "accept_lengths": cur_accept_lengths_tree, "cascade_accept_lengths": cur_cascade_accept_lengths_tree, "generate_speed": generate_speed})
            else:
                choices.append({"index": sample_id, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                                "accept_lengths": cur_accept_lengths_tree, "generate_speed": generate_speed})

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": task_id,
                "category": "humaneval",
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
            
            total_new_tokens += new_token
            total_time += decode_time
            progress_bar.update(1)

    result = None
    pred_dir =  os.path.dirname(answer_file)
    pred_filename = f"{pred_dir}/humaneval_predictions.jsonl"
    write_jsonl(pred_filename, eval_samples)
    print("Evaluating...")
    result = entry_point(problem_file=question_file, sample_file=pred_filename)
    print(result)
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))
    print("#Generate latency: ", total_new_tokens / total_time)



