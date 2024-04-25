CUDA_VISIBLE_DEVICES=0 python cli_demo.py \
    --model_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/llama3_v4_insuranceQA \
    --template llama3 \
    --finetuning_type freeze \
    # --adapter_name_or_path ../../saves/LLaMA2-7B/lora/sft \