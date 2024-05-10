CUDA_VISIBLE_DEVICES=3 python /home/zhangmin/toby/IBA_Project_24spr/train_LLM/web_demo.py \
    --template llama3 \
    --model_name_or_path /home/zhangmin/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct \
    # --model_name_or_path /home/zhangmin/toby/IBA_Project_24spr/exp_model/InsuranceGPT \
    # --adapter_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/igpt_v1_rlhf_travel_insurance \