Model_Path=models/Meta-Llama-3-8B-Instruct-W4A8-QoQ
Eagle_Path=models/EAGLE-LLaMA3-Instruct-8B-on-W4A8-QoQ
Model_id="llama-3-8b-instruct"
Bench_name="gsm8k"

python3 evaluation/inference_eagle_w4a8_qoq_chn.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}/w4a8-qoq/eagle \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --eagle-num-iter 6 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 60 \
    --max-new-tokens 256