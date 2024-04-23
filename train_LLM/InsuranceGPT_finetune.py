import os
import json
from llmtuner import run_exp, ChatModel
import torch

torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


file_path = '/home/zhangmin/toby/IBA_Project_24spr/data/dataset_info.json'
with open(file_path, 'r') as file:
    data = json.load(file)
data['InsuranceCorpus'] = {'hf_hub_url': 'Ddream-ai/InsuranceCorpus',
                           "columns": {
                               "prompt": "咨询",
                               "response": "回复"}}
data['US_Airline_Sentiment'] = {'hf_hub_url': 'Shayanvsf/US_Airline_Sentiment',
                                "columns": {
                                    "query": "text",
                                    "prompt": "你现在需要判断这个评论的情感是积极还是消极的，如果是积极的，请输出1，如果是消极的，请输出0",
                                    "response": "airline_sentiment"}}

with open(file_path, 'w') as file:
    json.dump(data, file, indent=4)

# chatglm2_6b --> InsuranceGPT
# qwen --> InsuranceGPT
finetune_config1 = dict(
    stage="sft",
    do_train=True,
    model_name_or_path="Qwen/Qwen1.5-7B-Chat",
    template="qwen",
    dataset_dir="/home/zhangmin/toby/IBA_Project_24spr/data",
    dataset="sentiment",
    finetuning_type="lora",
    output_dir="/home/zhangmin/toby/IBA_Project_24spr/saves/insurance_qwen_7b_sentiment",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_steps=100,
    learning_rate=0.0002,
    num_train_epochs=5,
    max_grad_norm=0.5,
    # fp16=True,
    overwrite_output_dir=True,
    # quantization_bit=8,
    # upcast_layernorm=True,
)

# sentiment analysis
finetune_config2 = dict(
    stage="sft",
    do_train=True,
    model_name_or_path="/home/zhangmin/.kaggle/chatglm2-6b/",
    template="chatglm2",
    dataset_dir="/home/zhangmin/toby/IBA_Project_24spr/data",
    dataset="US_Airline_Sentiment",
    finetuning_type="lora",
    output_dir="/home/zhangmin/toby/IBA_Project_24spr/saves/insurance_glm2_6b_sentiment",
    adapter_name_or_path="/home/zhangmin/toby/IBA_Project_24spr/saves/insurance_glm2_6b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_steps=100,
    learning_rate=0.0002,
    num_train_epochs=5,
    max_grad_norm=0.5,
    fp16=True,
    overwrite_output_dir=True,
    quantization_bit=8,
    upcast_layernorm=True,
)
# run_exp(finetune_config1)
run_exp()
