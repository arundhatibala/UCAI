from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import RewardConfig, RewardTrainer, is_xpu_available
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

def formatting_func(example, tokenizer):
  kwargs = {"padding": "max_length",
            "truncation": True,
            "max_length": 512,
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
        bnb_4bit_use_double_quant=True,
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
        trust_remote_code=True,
        num_labels=1,
        device_map={"":0},
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Step 2: Load the dataset and pre-process it
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    questions=pd.read_csv("preference_training_positive.csv", index_col=None)
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
        output_dir="tiny_output",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="steps",
        save_strategy="epoch",
        max_length=512,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cpu=True,
    )

    # Step 5: Define the Trainer
    model.config.pad_token_id = model.config.eos_token_id

    # Step 5: Define the Trainer
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=formatted_train,
        eval_dataset=formatted_test,
        peft_config=peft_params,
    )

    trainer.train()

    trainer.save_model("reward_llama_good")


if __name__ == "__main__":
    main()
