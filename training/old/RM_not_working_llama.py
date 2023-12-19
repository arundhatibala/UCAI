from dataclasses import dataclass, field
from typing import Optional

from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import RewardConfig, RewardTrainer, is_xpu_available
import torch
import torch.nn as nn

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

peft_params = LoraConfig(
lora_alpha=16,
lora_dropout=0.1,
r=64,
bias="none",
task_type="CAUSAL_LM",
)

access_token="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"


base_model="meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        token=access_token,
        num_labels=1,
        quantization_config=quant_config,
        device_map={"": 0}
    )
model.config.use_cache = False
model.config.pretraining_tp = 1


tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["response_a"], examples["response_b"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

dataset = load_dataset("andersonbcdefg/redteaming_eval_pairwise")
dataset = dataset["train"].train_test_split(test_size=0.2)


train_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)

dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= 512
    and len(x["input_ids_rejected"]) <= 512
)

### Loading the TRL reward trainer and training the trainer
training_args = RewardConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=512
)

# Step 5: Define the Trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset['train'],
    eval_dataset=train_dataset['test'],
    peft_config=peft_params,
)

trainer.train()

trainer.save_model("test_reward")

