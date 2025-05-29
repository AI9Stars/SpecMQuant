Model_Path=meta-llama/Meta-Llama-3-8B-Instruct
Quant_Path=quant_models/Meta-Llama-3-8B-Instruct-W4A16-g128
Output_Path=models/Meta-Llama-3-8B-Instruct-W4A16-g128


python model_convert/convert_w4a16.py \
    --model-path $Model_Path \
    --quant-path $Quant_Path \
    --output-path $Output_Path 