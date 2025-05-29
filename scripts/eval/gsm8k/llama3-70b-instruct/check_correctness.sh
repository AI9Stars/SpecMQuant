question_file="data/gsm8k/gsm8k/main"

echo "gsm8k w8a8"
sample_file="data/gsm8k/model_answer/llama-3-70b-instruct/w8a8/baseline.jsonl"
python evaluation/gsm8k/check_correctness.py \
    --question-file $question_file \
    --sample-file $sample_file

echo "gsm8k w4a16"
sample_file="data/gsm8k/model_answer/llama-3-70b-instruct/w4a16/baseline.jsonl"
python evaluation/gsm8k/check_correctness.py \
    --question-file $question_file \
    --sample-file $sample_file

echo "gsm8k w4a8-qqq"
sample_file="data/gsm8k/model_answer/llama-3-70b-instruct/w4a8-qqq/baseline.jsonl"
python evaluation/gsm8k/check_correctness.py \
    --question-file $question_file \
    --sample-file $sample_file

echo "gsm8k w4a8-qqq-g128"
sample_file="data/gsm8k/model_answer/llama-3-70b-instruct/w4a8-qqq-g128/baseline.jsonl"
python evaluation/gsm8k/check_correctness.py \
    --question-file $question_file \
    --sample-file $sample_file

echo "gsm8k w4a8-qoq"
sample_file="data/gsm8k/model_answer/llama-3-70b-instruct/w4a8-qoq/baseline.jsonl"
python evaluation/gsm8k/check_correctness.py \
    --question-file $question_file \
    --sample-file $sample_file

echo "gsm8k w4a8-qoq-g128"
sample_file="data/gsm8k/model_answer/llama-3-70b-instruct/w4a8-qoq-g128/baseline.jsonl"
python evaluation/gsm8k/check_correctness.py \
    --question-file $question_file \
    --sample-file $sample_file