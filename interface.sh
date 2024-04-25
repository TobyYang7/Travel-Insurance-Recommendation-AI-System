CUDA_VISIBLE_DEVICES=0 python train_LLM/cli_demo.py \
    --model_name_or_path exp_model/InsuranceGPT \
    --template llama3 \
    --finetuning_type freeze \
    # --adapter_name_or_path ../../saves/LLaMA2-7B/lora/sft \