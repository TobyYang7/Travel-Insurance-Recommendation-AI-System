#!/bin/bash

DATASET_NAME="travel_insurance"
MODEL_NAME="igpt_travel_insurance"
MODEL_PATH="/home/zhangmin/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct"
# MODEL_PATH="/home/zhangmin/toby/IBA_Project_24spr/exp_model/InsuranceGPT"

# CUDA_VISIBLE_DEVICES=2 python run_exp.py \
deepspeed --num_gpus 4 run_exp.py \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --dataset_dir /home/zhangmin/toby/IBA_Project_24spr/data \
    --template 	llama3 \
    --finetuning_type freeze \
    --output_dir /home/zhangmin/toby/IBA_Project_24spr/saves/$MODEL_NAME \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 200 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --val_size 0.2 \
    --plot_loss \
    --fp16 \
    --load_best_model_at_end \
    --deepspeed  ds_z3_config.json \
    --ddp_timeout 180000000 \
