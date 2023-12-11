import os
import torch
from datasets import load_dataset
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

base_model="meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=access_token,
        quantization_config=quant_config,
    )
model.config.use_cache = False
model.config.pretraining_tp = 1

model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Fine-tuned model
new_model = "SL_test"

################################################################################
# CODE SCRIPT

dataset = load_dataset("csv", data_files="critique_revisions.csv", split="train")

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model.module,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=128,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)
trainer.train()