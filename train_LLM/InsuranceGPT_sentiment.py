from typing import List, Dict, Optional
import os
import datasets
import torch
from loguru import logger
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from torch.nn import DataParallel
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
import torch
from torch.utils.data import DataLoader
from peft import PeftModel
from util import checkpoint, PrintLossCallback, ModifiedTrainer, data_collator, tokenizer


torch.cuda.empty_cache()

# load data
print("Loading data...")
dataset = datasets.load_from_disk("/home/zhangmin/toby/IBA_Project_24spr/data/flight_dataset")
dataset = dataset.train_test_split(0.2, shuffle=True, seed=42)

train_data = dataset['train']
eval_data = dataset['test']

# load model
print("Loading model...")
resume_from_checkpoint = None
print("Available devices: ", torch.cuda.device_count())
model_name = "THUDM/chatglm2-6b"
local_model_path = os.path.expanduser("~/.kaggle/chatglm2-6b/")
peft_model = "/home/zhangmin/toby/IBA_Project_24spr/saves/insurance_glm2_6b"
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    load_in_8bit=True,
    trust_remote_code=True,
    device_map='cuda:0'
)
model = PeftModel.from_pretrained(model, peft_model)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules,
    bias='none',
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
checkpoint(resume_from_checkpoint, model)


# train
print("Training...")
training_args = TrainingArguments(
    output_dir='/home/zhangmin/toby/IBA_Project_24spr/saves/insurance_glm2_6b_sentiment',
    logging_steps=500,
    # max_steps=10000,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    save_steps=500,
    fp16=True,
    # bf16=True,
    torch_compile=False,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    remove_unused_columns=False
)

trainer = ModifiedTrainer(
    model=model,
    args=training_args,             # Trainer args
    train_dataset=train_data,       # Training set
    eval_dataset=eval_data,         # Testing set
    data_collator=data_collator,    # Data Collator
)

trainer.train()
model.save_pretrained(training_args.output_dir)
