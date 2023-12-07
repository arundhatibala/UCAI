import torch
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from trl import RewardTrainer, SFTTrainer
from datasets import Dataset
import json
import pandas as pd
from transformers import Trainer, TrainingArguments
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

# model path for the Reward Model (eventually ours)
MODEL_PATH = "bigcode/tiny_starcoder_py"
DATA_PATH = "data/test.parquet" # don't know exactly what this is