#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

CUDA_VISIBLE_DEVICES=0 python /home/zhangmin/toby/IBA_Project_24spr/train_LLM/exp_model.py \
    --model_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/llama3_v4_insuranceQA \
    --adapter_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/llama3_v5_insuranceQA \
    --template llama3 \
    --finetuning_type lora \
    --export_dir  /home/zhangmin/toby/IBA_Project_24spr/exp_model/InsuranceGPT \
    --export_size 2 \
    --export_legacy_format False
