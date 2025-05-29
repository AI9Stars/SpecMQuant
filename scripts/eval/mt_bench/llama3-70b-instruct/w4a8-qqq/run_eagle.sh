Model_Path=models/Meta-Llama-3-70B-Instruct-W4A8-QQQ
Eagle_Path=models/EAGLE-LLaMA3-Instruct-70B-on-W4A8-QQQ
Model_id="llama-3-70b-instruct"
Bench_name="mt_bench"

python3 evaluation/inference_eagle_w4a8_qqq.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}/w4a8-qqq/eagle \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --eagle-num-iter 6 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 48