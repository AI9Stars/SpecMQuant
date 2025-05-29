Model_Path=models/Meta-Llama-3-70B-Instruct-W4A8-QoQ-g128
Eagle_Path=models/EAGLE-LLaMA3-Instruct-70B-on-W4A8-QoQ
Model_id="llama-3-70b-instruct"
Bench_name="human_eval"

python3 evaluation/inference_eagle_w4a8_qoq_group.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}/w4a8-qoq-g128/eagle \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --eagle-num-iter 6 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 48 \
    --max-new-tokens 512