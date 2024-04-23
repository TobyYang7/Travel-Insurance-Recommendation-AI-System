#!/bin/bash

deepspeed --num_gpus 4 InsuranceGPT_finetune.py \
    --deepspeed  ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen1.5-7B-Chat \
    --dataset identity,sentiment,InsuranceCorpus\
    --dataset_dir /home/zhangmin/toby/IBA_Project_24spr/data \
    --template qwen \
    --finetuning_type lora \
    --output_dir /home/zhangmin/toby/IBA_Project_24spr/saves/insurance_qwen_7b_sentiment \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --val_size 0.1 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --fp16
