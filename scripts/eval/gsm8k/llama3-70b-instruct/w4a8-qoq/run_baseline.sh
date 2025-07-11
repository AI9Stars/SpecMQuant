Model_Path=models/Meta-Llama-3-70B-Instruct-W4A8-QoQ
Model_id="llama-3-70b-instruct"
Bench_name="gsm8k"

python3 evaluation/inference_baseline_w4a8_qoq_chn.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}/w4a8-qoq/baseline \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --max-new-tokens 256
