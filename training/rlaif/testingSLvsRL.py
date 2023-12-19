from datasets import load_dataset
import json
import csv
import pandas as pd
from datasets import Dataset

questions = pd.read_csv("filtered_red_questions.csv")
testing=pd.DataFrame(questions, columns=["question"])
testing=testing[1002:]


testing=pd.DataFrame(testing, columns=["question"])
testing
print(testing)
dataset_testing = Dataset.from_pandas(testing)
print(dataset_testing)

# Supervised Learning
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForSequenceClassification,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer, AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch.nn as nn

# GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("\n\nUsing GPU:", torch.cuda.get_device_name(0), "\n")
else:
    device = torch.device("cpu")
    print("\n\nGPU not available, using CPU instead.")

# GOOD PRINCIPLES

supervised_model_path = "/home/marugan/UCAI/models/supervised_gpt2_good"
sl_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    supervised_model_path, local_files_only=True
)

config = PPOConfig(
    model_name="supervised_gpt2_good",
    learning_rate=1.41e-5,
    batch_size = 1,
)

rl_model_path = "/home/marugan/UCAI/models/RL_gpt2_good"
rl_model = AutoModelForSequenceClassification.from_pretrained(
    rl_model_path, local_files_only=True
)
rl_model.config.use_cache = False
rl_model.config.pretraining_tp = 1

rl_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token
'''
compute_dtype = getattr(torch, "float16")

rl_model.config.use_cache = False
rl_model.config.pretraining_tp = 1

rl_model.to(device)
#model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token

ppo_trainer = PPOTrainer(
    model=sl_model,
    config=config,
    dataset=dataset_testing,
    tokenizer=tokenizer,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 256,
    "batch_size": 1
}

for epoch, batch in enumerate(ppo_trainer.dataset):
    if epoch == 2:
        break
    print(batch["question"])
'''
# BAD PRINCIPLES

def ask_prompt(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs=inputs.to(device)
    generation_output = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=100, min_new_tokens=10)

    response = tokenizer.batch_decode(generation_output['sequences'], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


print("------- THIS IS THE TESTING OF SL vs RL -------")
sampled_questions = testing['question'].sample(5).tolist()
for n in sampled_questions:
    print("----- SL model -----")
    print(n)
    answer = ask_prompt(sl_model,tokenizer,n,device)
    print(answer)
    print("----- RL model -----")
    print(n)
    answer = ask_prompt(rl_model,tokenizer,n,device)
    print(answer)
    
print("------- THIS IS THE TESTING OF GOOD vs EVIL -------")