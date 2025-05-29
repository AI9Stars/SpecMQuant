Model_Path=meta-llama/Meta-Llama-3-8B-Instruct
Eagle_Path=models/EAGLE-LLaMA3-Instruct-8B
Model_id="llama-3-8b-instruct"
Bench_name="mt_bench"

python3 evaluation/inference_eagle.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}/fp16/eagle \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --eagle-num-iter 6 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 60