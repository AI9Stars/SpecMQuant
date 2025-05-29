
tokenizer_path="/home/ydzhang/checkpoints/meta-llama/Meta-Llama-3-70B-Instruct"
base_file="data/spec_bench/model_answer/llama-3-70b-instruct/w4a16/baseline.jsonl"

echo "vailla spec"
spec_file="data/spec_bench/model_answer/llama-3-70b-instruct/w4a16/spec.jsonl"
python evaluation/spec_bench/speed.py \
    --file-path $spec_file \
    --base-path $base_file \
    --tokenizer-path $tokenizer_path

echo "eagle-2"
spec_file="data/spec_bench/model_answer/llama-3-70b-instruct/w4a16/eagle.jsonl"
python evaluation/spec_bench/speed.py \
    --file-path $spec_file \
    --base-path $base_file \
    --tokenizer-path $tokenizer_path

echo "hierspec"
spec_file=data/spec_bench/model_answer/llama-3-70b-instruct/w4a16/hierspec.jsonl
python evaluation/spec_bench/speed.py \
    --file-path $spec_file \
    --base-path $base_file \
    --tokenizer-path $tokenizer_path