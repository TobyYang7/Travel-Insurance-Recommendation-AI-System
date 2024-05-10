from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.utils.data import DataLoader
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from torch.nn import DataParallel
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
from datasets import load_dataset
from loguru import logger
import datasets
from typing import List, Dict, Optional
import os
import json
from llmtuner import run_exp, ChatModel
import torch
from util import generate_text
import jsonlines

torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
InsuranceGPT_path = "/home/zhangmin/toby/IBA_Project_24spr/saves/insurance_glm2_6b"
InsuranceGPT_sentiment_path = "/home/zhangmin/toby/IBA_Project_24spr/saves/insurance_glm2_6b_sentiment"
glm2_path = os.path.expanduser("~/.kaggle/chatglm2-6b/")
data_path = "/home/zhangmin/toby/IBA_Project_24spr/data/flight_dataset.jsonl"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=True,
)


glm2_path = os.path.expanduser("~/.kaggle/chatglm2-6b/")
tokenizer = AutoTokenizer.from_pretrained(glm2_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(glm2_path, trust_remote_code=True, device_map="auto")
InsuranceGPT = PeftModel.from_pretrained(base_model, InsuranceGPT_path)
InsuranceGPT_sentiment = PeftModel.from_pretrained(base_model, InsuranceGPT_sentiment_path)


base_model.print_trainable_parameters()
InsuranceGPT.print_trainable_parameters()
# base_model = base_model.eval()
# InsuranceGPT = InsuranceGPT.eval()
# InsuranceGPT_sentiment = InsuranceGPT_sentiment.eval()


query = "hello"
print(generate_text(base_model, tokenizer, query))
print(generate_text(InsuranceGPT, tokenizer, query))
print(generate_text(InsuranceGPT_sentiment, tokenizer, query))
