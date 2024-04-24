#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python InsuranceGPT_exp.py \
    --stage sft \
    --model_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/llama3_v1_mix \
    --dataset travel_insurance_test \
    --dataset_dir /home/zhangmin/toby/IBA_Project_24spr/data \
    --template llama3 \
    --finetuning_type freeze \
    --output_dir /home/zhangmin/toby/IBA_Project_24spr/exp/llama3_v1_mix/travel_insurance_test \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 1 \
    --max_new_tokens 200 \
    --top_p 0.8 \
    --temperature 0.8 \
    --do_predict \
    --predict_with_generate \
    --max_sample 1000 \