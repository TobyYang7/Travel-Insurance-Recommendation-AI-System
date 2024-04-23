---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: Qwen/Qwen1.5-7B-Chat
model-index:
- name: InsuranceGPT
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# InsuranceGPT

This model is a fine-tuned version of [Qwen/Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat) on the identity and the QA_sentiment datasets.
It achieves the following results on the evaluation set:
- Loss: 1.6065

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- total_eval_batch_size: 4
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 20
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 2.0331        | 0.44  | 100  | 1.7115          |
| 2.0417        | 0.88  | 200  | 1.6614          |
| 1.9037        | 1.32  | 300  | 1.6364          |
| 1.8929        | 1.77  | 400  | 1.6176          |
| 1.6542        | 2.21  | 500  | 1.6082          |
| 1.7774        | 2.65  | 600  | 1.6076          |


### Framework versions

- PEFT 0.10.1.dev0
- Transformers 4.40.0.dev0
- Pytorch 2.2.2+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2