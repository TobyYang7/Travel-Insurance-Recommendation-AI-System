#!/bin/bash

 export HF_HUB_ENABLE_HF_TRANSFER=1 
 export HF_ENDPOINT=https://hf-mirror.com
# MODEL_NAME
DATASET_NAME1="travel_insurance_4_label_test"
DATASET_NAME2="travel_insurance_test"
DATASET_NAME3="sentiment_test"
DATASET_NAME4="insuranceQA"
MODEL_NAME="igpt_v1_rlhf_travel_insurance"
MODEL_PATH="../saves/igpt_v1_rlhf_travel_insurance"
# MODEL_PATH="../exp_model/InsuranceGPT"
# MODEL_NAME="llama3"
# MODEL_PATH="/home/zhangmin/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct"

CUDA_VISIBLE_DEVICES=2  python run_exp.py \
    --model_name_or_path ../saves/igpt_v1_rlhf \
    --dataset $DATASET_NAME3 \
    --dataset_dir ../data \
    --template llama3 \
    --output_dir ../saves/exp/$MODEL_NAME/$DATASET_NAME3 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 8192 \
    --per_device_train_batch_size 1 \
    --max_new_tokens 200 \
    --top_p 0.5 \
    --temperature 0.5 \
    --do_predict \
    --predict_with_generate \
    --per_device_eval_batch_size 8 \
    --max_samples 3000 \
    --adapter_name_or_path $MODEL_PATH \
    # --split test
