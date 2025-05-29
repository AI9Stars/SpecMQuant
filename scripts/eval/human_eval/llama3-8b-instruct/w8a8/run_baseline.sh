Model_Path=models/Meta-Llama-3-8B-Instruct-W8A8
Model_id="llama-3-8b-instruct"
Bench_name="human_eval"

python3 evaluation/inference_baseline_w8a8.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}/w8a8/baseline \
    --memory-limit 0.8 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --max-new-tokens 512
