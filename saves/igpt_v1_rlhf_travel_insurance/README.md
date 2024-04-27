---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /home/zhangmin/toby/IBA_Project_24spr/saves/igpt_v1_rlhf
model-index:
- name: igpt_v1_rlhf_travel_insurance
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# igpt_v1_rlhf_travel_insurance

This model is a fine-tuned version of [/home/zhangmin/toby/IBA_Project_24spr/saves/igpt_v1_rlhf](https://huggingface.co//home/zhangmin/toby/IBA_Project_24spr/saves/igpt_v1_rlhf) on the travel_insurance dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1539

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
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- total_eval_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 4.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.1291        | 1.0   | 50   | 0.1607          |
| 0.1504        | 2.0   | 100  | 0.1546          |
| 0.1432        | 3.0   | 150  | 0.1544          |
| 0.1117        | 4.0   | 200  | 0.1539          |


### Framework versions

- PEFT 0.10.1.dev0
- Transformers 4.40.0.dev0
- Pytorch 2.2.2+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2