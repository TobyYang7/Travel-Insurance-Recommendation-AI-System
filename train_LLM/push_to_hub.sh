CUDA_VISIBLE_DEVICES=1 python run_exp.py \
                            --push_to_hub \
                            --model_name_or_path /home/zhangmin/toby/IBA_Project_24spr/saves/llama3_v2_insuranceQA \
                            --output_dir push_to_hub \
                            # --push_to_hub_model_id