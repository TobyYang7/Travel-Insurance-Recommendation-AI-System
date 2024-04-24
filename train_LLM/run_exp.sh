#!/bin/bash

# MODEL_NAME
DATASET_NAME="sentiment_test,travel_insurance_test"
MODEL_NAME="llama3_v3_insuranceQA"

CUDA_VISIBLE_DEVICES=1 python run_exp.py \
    --stage sft \
    --model_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/llama3_v2_insuranceQA \
    --dataset $DATASET_NAME \
    --dataset_dir /home/zhangmin/toby/IBA_Project_24spr/data \
    --template llama3 \
    # --output_dir /home/zhangmin/toby/IBA_Project_24spr/saves/exp/$MODEL_NAME/$DATASET_NAME \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 1 \
    --max_new_tokens 200 \
    --top_p 0.5 \
    --temperature 0.5 \
    --do_predict \
    --predict_with_generate \
    --adapter_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/llama3_v3_insuranceQA \
    --per_device_eval_batch_size 16 \
    --max_samples 3000 \