Model_Path=models/Meta-Llama-3-8B-Instruct-W4A16-g128-Rot
Model_id="llama-3-8b-instruct"
Bench_name="human_eval"

python3 evaluation/inference_baseline_w4a16_gptq_marlin.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}/w4a16/baseline \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --max-new-tokens 512


##### Uncomment the following lines to run the baseline without rotation
# Model_Path=models/Meta-Llama-3-8B-Instruct-W4A16-g128
# Model_id="llama-3-8b-instruct"
# Bench_name="human_eval"

# python3 evaluation/inference_baseline_w4a16_gptq_marlin.py \
#     --model-path $Model_Path \
#     --cuda-graph \
#     --model-id ${Model_id}/w4a16/baseline \
#     --memory-limit 0.80 \
#     --bench-name $Bench_name \
#     --dtype "float16" \
#     --max-new-tokens 512