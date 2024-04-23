import random
from retrying import retry
import openai
from tqdm import tqdm
import jsonlines
from transformers import TrainerCallback
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
openai.api_base = "https://api.ai-gaochao.cn/v1"


def checkpoint(resume_from_checkpoint, model):
    if resume_from_checkpoint is not None:
        checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, 'flight_review_adapter_model.bin'
            )
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            logger.info(f'Restarting from {checkpoint_name}')
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info(f'Checkpoint {checkpoint_name} not found')


class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"[{state.global_step}/{state.max_steps} {state.epoch:.2f}/{args.num_train_epochs}] - Step {state.global_step}: Training Loss = {logs.get('loss', 'N/A')}")


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def prediction_step(self, model: torch.nn.Module, inputs, prediction_loss_only: bool, ignore_keys=None):
        with torch.no_grad():
            res = model(
                input_ids=inputs["input_ids"].to(model.device),
                labels=inputs["labels"].to(model.device),
            ).loss
        return (res, None, None)

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cuda") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "flight_review_adapter_model.bin"))


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [tokenizer.pad_token_id] * (seq_len - 1) + ids[(seq_len - 1):] + [tokenizer.pad_token_id] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def generate_text(model, tokenizer, input_text, temperature=0.5, top_p=0.1,  n=1):
    """
    Generates text based on the input text using the specified model and tokenizer.

    Parameters:
        model (torch.nn.Module): The pre-trained language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding the text.
        input_text (str): The text to generate from.
        temperature (float): Controls the randomness of the predictions by scaling the logits before applying softmax.
        top_p (float): The cumulative probability for top-p-filtering.
        frequency_penalty (float): The penalty for frequency of tokens in the generated text.
        presence_penalty (float): The penalty for presence of tokens in the generated text.
        n (int): The number of sequences to generate.

    Returns:
        str: The generated text.
    """

    tokenizer.padding_side = "left"
    if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

    # Generate the sequence
    output_ids = model.generate(
        input_ids,
        max_length=200,  # Assuming max_length is fixed or you can make it a parameter
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,  # Assuming you want to avoid repeating n-grams, adjust as needed
        do_sample=True
    )

    # Decode and return the generated text
    generated_texts = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]
    return generated_texts[0] if n == 1 else generated_texts


def score(model, path, max_length=200):
    count = 0
    total = 0
    with jsonlines.open(path) as reader:
        for obj in tqdm(reader, desc=f"Scoring {str(model)}", dynamic_ncols=True, smoothing=0.1):
            total += 1
            ground_truth = obj['target']
            context = obj['context']
            answer = model.chat([{"role": "user", "content": ""}], system=context)
            if ground_truth in answer:
                count += 1
            if total == max_length:
                break
    return count / total if total > 0 else 0


class OpenAIGPT:
    def __init__(self, model_name="gpt-3.5-turbo", keys_path=None):
        self.model_name = model_name
        with open(keys_path, encoding="utf-8", mode="r") as fr:
            self.keys = [line.strip() for line in fr if len(line.strip()) >= 4]

    def __post_process(self, response):
        return response["choices"][0]["message"]["content"]

    @retry(wait_fixed=300, stop_max_attempt_number=50)
    def __call__(self, message):
        if message is None or message == "":
            return False, "Your input is empty."

        # current_key = random.choice(self.keys)
        current_key = self.keys[0] if len(self.keys) == 1 else random.choice(self.keys)
        openai.api_key = current_key
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": message}],
            temperature=0.3,
            top_p=0.1,
            frequency_penalty=0.6,
            presence_penalty=0.6,
            n=1,
        )
        return self.__post_process(response)
