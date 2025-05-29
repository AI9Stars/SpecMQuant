
tokenizer_path="meta-llama/Meta-Llama-3-8B-Instruct"
base_file="data/mt_bench/model_answer/llama-3-8b-instruct/fp16/baseline.jsonl"

echo "fp16-eagle"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/fp16/eagle.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path

echo "w8a8"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/w8a8/baseline.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path

echo "w8a8-eagle"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/w8a8/eagle.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path

echo "w4a16"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/w4a16/baseline.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path

echo "w4a16-eagle"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/w4a16/eagle.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path

echo "w4a8-qqq"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/w4a8-qqq/baseline.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path

echo "w4a8-qqq-eagle"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/w4a8-qqq/eagle.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path


echo "w4a8-qqq-g128"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/w4a8-qqq-g128/baseline.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path

echo "w4a8-qqq-g128-eagle"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/w4a8-qqq-g128/eagle.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path

echo "w4a8-qoq"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/w4a8-qoq/baseline.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path

echo "w4a8-qoq-eagle"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/w4a8-qoq/eagle.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path

echo "w4a8-qoq-g128"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/w4a8-qoq-g128/baseline.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path

echo "w4a8-qoq-g128-eagle"
spec_file="data/mt_bench/model_answer/llama-3-8b-instruct/w4a8-qoq-g128/eagle.jsonl"
python evaluation/spec_bench/speed_mt_bench.py \
    --file-path $spec_file \
    --base-path $base_file \
    --checkpoint-path $tokenizer_path
