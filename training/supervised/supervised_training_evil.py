import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split


################################################################################
# MODEL

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

access_token="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"

base_model="gpt2"

model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"":3}
    )
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

################################################################################
# CODE SCRIPT

# Load your CSV file
df = pd.read_csv("critique_revisions_evil.csv")

# Split the data into training and test sets (adjust test_size as needed)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert pandas DataFrames to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

print(train_dataset)
print(test_dataset)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    logging_steps=500,
    learning_rate=2e-4,
    weight_decay=0.001,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)
trainer.train()

trainer.save_model("../../models/supervised_gpt2_evil")