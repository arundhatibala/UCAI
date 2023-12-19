from datasets import load_dataset
import json
import csv
import pandas as pd
from datasets import Dataset
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
import sys

dataset = pd.read_csv("../../prompts/filtered_red_questions.csv")
dataset=pd.DataFrame(dataset, columns=["question"])


device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")


# GOOD PRINCIPLES
good_model_path = "/home/marugan/UCAI/models/RL_gpt2_good"
rl_model_good = AutoModelForCausalLM.from_pretrained(
    good_model_path, 
    local_files_only=True,
    device_map={"":0}
)
rl_model_good.config.pad_token_id=rl_model_good.config.eos_token_id
rl_model_good.config.use_cache = False
rl_model_good.config.pretraining_tp = 1

tokenizer_good = AutoTokenizer.from_pretrained(good_model_path, local_files_only=True)

tokenizer_good.pad_token = tokenizer_good.eos_token


evil_model_path = "/home/marugan/UCAI/models/RL_gpt2_evil"
rl_model_evil = AutoModelForCausalLM.from_pretrained(
    evil_model_path, 
    local_files_only=True,
    device_map={"":1}
)
rl_model_evil.config.pad_token_id=rl_model_evil.config.eos_token_id

tokenizer_evil = AutoTokenizer.from_pretrained(evil_model_path, local_files_only=True)
tokenizer_evil.pad_token = tokenizer_evil.eos_token



def ask_prompt(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs=inputs.to(device)
    generation_output = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=100, min_new_tokens=10)

    response = tokenizer.batch_decode(generation_output['sequences'], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


sampled_questions = dataset['question'].sample(5).tolist()
for question in sampled_questions:
    print(f"____________ NEW QUESTION _____________")
    print("QUESTION: " + question + "\n\n")
    
    print("----- RL model good -----\n")
    answer = ask_prompt(rl_model_good,tokenizer_good,question,device0) 
    print("ANSWER: " + answer)

    print("\n----- RL model evil----\n")
    answer = ask_prompt(rl_model_evil,tokenizer_evil,question,device1)
    print("ANSWER: " + answer)
    print("//////////////////////////////////////////")