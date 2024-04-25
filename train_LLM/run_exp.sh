#!/bin/bash

# MODEL_NAME
DATASET_NAME="travel_insurance_4_label_test"
MODEL_NAME="InsuranceGPT_v3"
MODEL_PATH="../exp_model/InsuranceGPT"
# MODEL_NAME="llama3"
# MODEL_PATH="/home/zhangmin/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct"

CUDA_VISIBLE_DEVICES=2 python run_exp.py \
    --stage sft \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --dataset_dir ../data \
    --template llama3 \
    --output_dir ../saves/exp/$MODEL_NAME/$DATASET_NAME \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 1 \
    --max_new_tokens 200 \
    --top_p 0.8 \
    --temperature 0.8 \
    --do_predict \
    --predict_with_generate \
    --per_device_eval_batch_size 8 \
    --max_samples 3000 \
    --adapter_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/llama3_v6_insuranceQA_lora_history \