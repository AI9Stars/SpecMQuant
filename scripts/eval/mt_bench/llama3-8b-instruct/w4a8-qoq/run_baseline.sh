Model_Path=models/Meta-Llama-3-8B-Instruct-W4A8-QoQ
Model_id="llama-3-8b-instruct"
Bench_name="mt_bench"

python3 evaluation/inference_baseline_w4a8_qoq_chn.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}/w4a8-qoq/baseline \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16"
