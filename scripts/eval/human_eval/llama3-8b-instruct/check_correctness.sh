Question_file=data/human_eval/HumanEval.jsonl.gz 

echo "human_eval fp16"
Sample_file=data/human_eval/model_answer/llama-3-8b-instruct/fp16/baseline/humaneval_predictions.jsonl
python evaluation/humaneval/check_correctness.py \
    --question-file $Question_file \
    --sample-file $Sample_file

echo "human_eval w8a8"
Sample_file=data/human_eval/model_answer/llama-3-8b-instruct/w8a8/baseline/humaneval_predictions.jsonl
python evaluation/humaneval/check_correctness.py \
    --question-file $Question_file \
    --sample-file $Sample_file

echo "human_eval w4a16"
Sample_file=data/human_eval/model_answer/llama-3-8b-instruct/w4a16/baseline/humaneval_predictions.jsonl
python evaluation/humaneval/check_correctness.py \
    --question-file $Question_file \
    --sample-file $Sample_file

echo "human_eval w4a8-qqq"
Sample_file=data/human_eval/model_answer/llama-3-8b-instruct/w4a8-qqq/baseline/humaneval_predictions.jsonl
python evaluation/humaneval/check_correctness.py \
    --question-file $Question_file \
    --sample-file $Sample_file

echo "human_eval w4a8-qqq-g128"
Sample_file=data/human_eval/model_answer/llama-3-8b-instruct/w4a8-qqq-g128/baseline/humaneval_predictions.jsonl
python evaluation/humaneval/check_correctness.py \
    --question-file $Question_file \
    --sample-file $Sample_file

echo "human_eval w4a8-qoq"
Sample_file=data/human_eval/model_answer/llama-3-8b-instruct/w4a8-qoq/baseline/humaneval_predictions.jsonl
python evaluation/humaneval/check_correctness.py \
    --question-file $Question_file \
    --sample-file $Sample_file

echo "human_eval w4a8-qoq-g128"
Sample_file=data/human_eval/model_answer/llama-3-8b-instruct/w4a8-qoq-g128/baseline/humaneval_predictions.jsonl
python evaluation/humaneval/check_correctness.py \
    --question-file $Question_file \
    --sample-file $Sample_file