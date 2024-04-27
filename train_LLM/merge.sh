#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

CUDA_VISIBLE_DEVICES=0 python /home/zhangmin/toby/IBA_Project_24spr/train_LLM/exp_model.py \
    --model_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/igpt_v1_rlhf \
    --adapter_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/igpt_v1_rlhf_travel_insurance \
    --template llama3 \
    --export_dir  /home/zhangmin/toby/IBA_Project_24spr/exp_model/InsuranceGPT \
    --export_legacy_format True \
    --export_size 20 \
