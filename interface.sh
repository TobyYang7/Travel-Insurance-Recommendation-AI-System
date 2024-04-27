CUDA_VISIBLE_DEVICES=0 python train_LLM/cli_demo.py \
    --model_name_or_path exp_model/InsuranceGPT \
    --template llama3 \
    --temperature 0.8 \
    # --adapter_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/igpt_v2_gpt4 \