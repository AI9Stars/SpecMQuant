import argparse
import torch
from fastchat.utils import str_to_torch_dtype
from evaluation.spec_bench.eval import run_eval as run_eval_spec_bench
from evaluation.gsm8k.eval import run_eval as run_eval_gsm8k
from evaluation.humaneval.eval import run_eval as run_eval_humaneval
from transformers import AutoTokenizer, AutoConfig
from llamacu.speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle


def eagle_w4a16_forward(inputs, model, tokenizer, max_new_tokens, max_length, teminators):
    if isinstance(inputs, torch.Tensor):
        input_ids = inputs.int()
    else:
        input_ids = inputs.input_ids.int()

    prefill_length = len(input_ids[0])
    max_new_tokens = min(max_new_tokens, max_length - prefill_length)
    
    # generate
    output_ids, accept_length_list, model_step, decode_time, *draft_prefill_latency = model.generate(
        input_ids=input_ids,
        generation_length=max_new_tokens,
        teminators=teminators,
    )

    new_token = len(output_ids)
    return output_ids, new_token, model_step, accept_length_list, decode_time, draft_prefill_latency[0] if draft_prefill_latency else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eagle-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
    )
    parser.add_argument("--model-id", type=str, default="baseline-llama-3-70b-fp16")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="spec_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=100000,
        help="The maximum length of the model input length.",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=4096,
        help="The chunk length of the model prefill.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--eagle-num-iter",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--eagle-topk-per-iter",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--eagle-tree-size",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--draft-prefill-sep",
        action="store_true",
        help="Draft prefill separation for draft latency",
    )
    parser.add_argument(
        "--quant-rotation",
        action="store_true",
        help="quatized model with rotation",
    )


    args = parser.parse_args()

    if args.bench_name == "gsm8k":
        question_file = f"data/gsm8k/gsm8k/main"
    elif args.bench_name == "human_eval":
        question_file =  f"data/human_eval/HumanEval.jsonl.gz"
    else:
        question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        if args.bench_name == "human_eval":
            answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}/model_output.jsonl"
        else:
            answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")
    
    config = AutoConfig.from_pretrained(args.model_path)
    max_length = min(args.max_length, config.max_position_embeddings)
    chunk_length = min(args.chunk_length, config.max_position_embeddings)

    model = W4A16GPTQMarlinLLM_with_eagle(
        base_path=args.model_path,
        eagle_path=args.eagle_path,
        memory_limit=args.memory_limit,
        chunk_length=chunk_length,
        dtype=str_to_torch_dtype(args.dtype),
        cuda_graph=args.cuda_graph,
        num_iter=args.eagle_num_iter,
        topk_per_iter=args.eagle_topk_per_iter,
        tree_size=args.eagle_tree_size,
        draft_prefill_sep=args.draft_prefill_sep,
        rotation=args.quant_rotation,
    )
    model.init_storage()
    model.load_from_hf()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if "llama-3" in args.model_id.lower() or "llama_3" in args.model_id.lower() or "llama3" in args.model_id.lower():
        teminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        teminators = [tokenizer.eos_token_id]

    if args.bench_name == "mt_bench" or args.bench_name == "spec_bench":
        run_eval = run_eval_spec_bench
    elif args.bench_name == "gsm8k":
        run_eval = run_eval_gsm8k
    elif args.bench_name == "human_eval":
        run_eval = run_eval_humaneval
    else:
        raise ValueError(f"Unknown benchmark name: {args.bench_name}")

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=eagle_w4a16_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        max_length=max_length,
        num_choices=args.num_choices,
        teminators=teminators,
    )
