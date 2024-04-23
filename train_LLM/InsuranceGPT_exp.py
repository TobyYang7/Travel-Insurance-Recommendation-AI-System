from llmtuner import run_exp
import os
import json
from llmtuner import run_exp, ChatModel
import torch

torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

exp1_config = {
    "model_name_or_path": "Qwen/Qwen1.5-7B-Chat",
    "template": "qwen",
    "dataset_dir": "/home/zhangmin/toby/IBA_Project_24spr/data",
    "dataset": "sentiment_test",
    # "cutoff_len": 1024,
    "per_device_train_batch_size": 2,
    "max_new_tokens": 200,
    "top_p": 0.8,
    "temperature": 0.8,
    "output_dir": "/home/zhangmin/toby/IBA_Project_24spr/saves/exp/insurance_qwen_7b_sentiment",
    "do_predict": True,
    "predict_with_generate": True,
}

run_exp(exp1_config)
