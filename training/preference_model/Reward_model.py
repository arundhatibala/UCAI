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
from datasets import Dataset
import pandas as pd

def formatting_func(example, tokenizer):
  kwargs = {"padding": "max_length",
            "truncation": True,
            "max_length": 256,
            "return_tensors": "pt"
            }

  prompt_plus_chosen_response = str(example["prompt"]) + "\n" + str(example["chosen"])
  prompt_plus_rejected_response = str(example["prompt"]) + "\n" + str(example["rejected"])

  # Then tokenize these modified fields.
  tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
  tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

  return {
      "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
      "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
  }

def main():

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
   
    base_model="../../models/supervised_gpt2_evil"

    model = AutoModelForSequenceClassification.from_pretrained(
       base_model,
       local_files_only=True,
       num_labels=1,
       device_map={"":0},
       )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.pad_token_id = model.config.eos_token_id
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    
    questions=pd.read_csv("preference_training_evil.csv", index_col=None)
    questions = questions.drop(columns=["Unnamed: 0"])

    dataset = Dataset.from_pandas(questions)
    dataset=dataset.train_test_split(test_size=0.2)

    formatted_train = dataset['train'].map(lambda x: formatting_func(x, tokenizer))

    formatted_test = dataset['test'].map(lambda x: formatting_func(x, tokenizer))

    formatted_train = formatted_train.filter(
        lambda x: len(x["input_ids_chosen"]) <= 512
        and len(x["input_ids_rejected"]) <= 512
    )

    formatted_test = formatted_test.filter(
        lambda x: len(x["input_ids_chosen"]) <= 512
        and len(x["input_ids_rejected"]) <= 512
    )

    ### Loading the TRL reward trainer and training the trainer
    training_args = RewardConfig(
        output_dir="output_gpt2_evil",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        logging_steps=50,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        max_length=512,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
    )

    # Step 5: Define the Trainer
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=formatted_train,
        eval_dataset=formatted_test,
    )

    trainer.train()

    trainer.save_model("reward_gpt2_evil")


if __name__ == "__main__":
    main()
