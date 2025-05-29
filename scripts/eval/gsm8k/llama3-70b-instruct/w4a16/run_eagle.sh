Model_Path=models/Meta-Llama-3-70B-Instruct-W4A16-g128-Rot
Eagle_Path=models/EAGLE-LLaMA3-Instruct-70B-on-W4A16-Rot
Model_id="llama-3-70b-instruct"
Bench_name="gsm8k"

python3 evaluation/inference_eagle_w4a16_gptq_marlin.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id ${Model_id}/w4a16/eagle \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --quant-rotation \
    --eagle-num-iter 6 \
    --eagle-topk-per-iter 10 \
    --eagle-tree-size 48 \
    --max-new-tokens 256


##### Uncomment the following lines to run the baseline without rotation
# Model_Path=models/Meta-Llama-3-70B-Instruct-W4A16-g128
# Eagle_Path=models/EAGLE-LLaMA3-Instruct-70B
# Model_id="llama-3-70b-instruct"
# Bench_name="gsm8k"

# python3 evaluation/inference_eagle_w4a16_gptq_marlin.py \
#     --model-path $Model_Path \
#     --eagle-path $Eagle_Path \
#     --cuda-graph \
#     --model-id ${Model_id}/w4a16/eagle \
#     --memory-limit 0.80 \
#     --bench-name $Bench_name \
#     --dtype "float16" \
#     --eagle-num-iter 6 \
#     --eagle-topk-per-iter 10 \
#     --eagle-tree-size 48 \
#     --max-new-tokens 256