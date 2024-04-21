from transformers import AutoTokenizer, AutoConfig
from tqdm.notebook import tqdm
import json
import pandas as pd
import datasets
from datasets import load_dataset
import os
import shutil

jsonl_path = "/home/zhangmin/toby/IBA_Project_24spr/data/flight_review.jsonl"
save_path = '/home/zhangmin/toby/IBA_Project_24spr/data/flight_review'


if os.path.exists(jsonl_path):
    os.remove(jsonl_path)

if os.path.exists(save_path):
    shutil.rmtree(save_path)

directory = "/home/zhangmin/toby/IBA_Project_24spr/data"
if not os.path.exists(directory):
    os.makedirs(directory)


try:
    # Attempt to read the CSV with the C engine
    dataset = pd.read_csv('/home/zhangmin/toby/IBA_Project_24spr/data/flight_review.csv', quotechar='"', escapechar="\\", engine='c', on_bad_lines='skip')
except Exception as e:
    print(f"Failed to parse with C engine: {e}")
    # If C engine fails, fall back to the Python engine
    dataset = pd.read_csv('/home/zhangmin/toby/IBA_Project_24spr/data/flight_review.csv', quotechar='"', escapechar="\\", engine='python', on_bad_lines='skip')


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length, skip_overlength=False, model_name='Qwen/Qwen1.5-7B-Chat'):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


# select 'text' and 'airline_sentiment'
dataset = dataset[['text', 'airline_sentiment']]
dataset['airline_sentiment'] = dataset['airline_sentiment'].replace({
    'negative': 'A. negative',
    'neutral': 'B. neutral',
    'positive': 'C. positive'
})
dataset['instruction'] = 'What is the sentiment of this tweet? Please choose an answer from {A. negative; B. neutral; C. positive}.'
dataset.columns = ['input', 'output', 'instruction']
dataset = datasets.Dataset.from_pandas(dataset)
dataset = dataset.shuffle(seed=42)


data_list = []
for item in dataset.to_pandas().itertuples():
    tmp = {}
    tmp["instruction"] = item.instruction
    tmp["input"] = item.input
    tmp["output"] = item.output
    data_list.append(tmp)

# save to a jsonl file
with open(jsonl_path, 'w') as f:
    for example in tqdm(data_list, desc="formatting.."):
        f.write(json.dumps(format_example(example)) + '\n')


model_name = "Qwen/Qwen1.5-7B-Chat"
# local_model_path = os.path.expanduser("~/.kaggle/chatglm2-6b/")
max_seq_length = 512
skip_overlength = True
dataset = datasets.Dataset.from_generator(
    lambda: read_jsonl(jsonl_path, max_seq_length, skip_overlength)
)
dataset.save_to_disk(save_path)
