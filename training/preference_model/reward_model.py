import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from trl import RewardTrainer, SFTTrainer, RewardConfig

def main():
    dataset = load_dataset("andersonbcdefg/redteaming_eval_pairwise")
    dataset = dataset["train"].train_test_split(test_size=0.2)

    formatted_dataset = dataset.map(formatting_func)

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.6", max_len=256)
    base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.6")

    ### Loading the TRL reward trainer and training the trainer
    training_args = RewardConfig(
            output_dir="../../models",
            num_train_epochs=1,
            gradient_accumulation_steps=1,
            save_strategy="steps",
            evaluation_strategy="steps",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            eval_accumulation_steps=1,
            eval_steps=100,
            save_steps=100,
            warmup_steps=50,
            learning_rate=1e-5,
            save_total_limit=1,
            no_cuda=True,
        )
    
    trainer = RewardTrainer(model=base_model,
                        tokenizer=tokenizer,
                        train_dataset=formatted_dataset['train'],
                        eval_dataset=formatted_dataset['test'],
                        args= training_args
                        )
    trainer.train()

    # save the reward model
    trainer.save_model("../../models/test_reward")

def formatting_func(tokenizer, examples):
    kwargs = {"padding": "max_length",
              "truncation": True,
              "max_length": 256,
              "return_tensors": "pt"
              }

    prompt_plus_chosen_response = examples["prompt"] + "\n" + examples["response_a"]
    prompt_plus_rejected_response = examples["prompt"] + "\n" + examples["response_b"]

    # Then tokenize these modified fields.
    tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
    }

if __name__ == "__main__":
    main()