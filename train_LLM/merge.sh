#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

CUDA_VISIBLE_DEVICES=0 python /home/zhangmin/toby/IBA_Project_24spr/train_LLM/exp_model.py \
    --model_name_or_path Qwen/Qwen1.5-7B-Chat \
    --adapter_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/InsuranceGPT \
    --template qwen \
    --finetuning_type lora \
    --export_dir  /home/zhangmin/toby/IBA_Project_24spr/exp_model/InsuranceGPT_v1 \
    --export_size 2 \
    --export_legacy_format False
