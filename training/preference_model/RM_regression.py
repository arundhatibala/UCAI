from dataclasses import dataclass, field
from typing import Optional

from accelerate import Accelerator
from datasets import load_dataset
#from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import RewardConfig, RewardTrainer, is_xpu_available
import torch

def main():

    tqdm.pandas()

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

    device=device = torch.device('cuda:0')

    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/opt-350m",
        trust_remote_code=True,
        num_labels=1,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.to(device)

    # Step 2: Load the dataset and pre-process it
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    dataset = load_dataset("andersonbcdefg/redteaming_eval_pairwise")
    dataset = dataset["train"].train_test_split(test_size=0.2)


    formatted_dataset = dataset.map(lambda x: formatting_func(x, tokenizer))

    dataset = formatted_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= 512
        and len(x["input_ids_rejected"]) <= 512
    )


    eval_dataset = None

    ### Loading the TRL reward trainer and training the trainer
    training_args = RewardConfig(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        max_length=512,
    )

    # Step 5: Define the Trainer
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=formatted_dataset['train'],
        eval_dataset=formatted_dataset['test'],
    )

    trainer.train()

    trainer.save_model("test_reward")

    ### Let's now take the saved model and generate a score



    model = AutoModelForSequenceClassification.from_pretrained("my_awesome_model/checkpoint-84/", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("my_awesome_model/checkpoint-84/", local_files_only=True)

    model.config.use_cache = False
    model.config.pretraining_tp = 1



    dataset = load_dataset("andersonbcdefg/redteaming_eval_pairwise")
    dataset=dataset['train']


    # usage with prompt
    prompt = dataset[0]["prompt"]
    example_prefered_response = dataset[0]["response_a"]
    example_unprefered_response = dataset[0]["response_b"]


    formatted_example = dataset.map(lambda x: formatting_func_2(example_prefered_response, tokenizer))
    print(formatted_example)

    answer = get_score(model, tokenizer, prompt, example_prefered_response)
    print(answer)


def formatting_func(example, tokenizer):
  kwargs = {"padding": "max_length",
            "truncation": True,
            "max_length": 256,
            "return_tensors": "pt"
            }

  prompt_plus_chosen_response = str(example["prompt"]) + "\n" + str(example["response_a"])
  prompt_plus_rejected_response = str(example["prompt"]) + "\n" + str(example["response_b"])

  # Then tokenize these modified fields.
  tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
  tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

  return {
      "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
      "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
  }

def get_score(model, tokenizer, prompt, response):

    instructions = tokenizer.encode_plus(prompt,
                                       response,
                                       padding="max_length",
                                       max_length=256,
                                       return_tensors="pt",
                                        truncation=True)
    with torch.no_grad():
        outputs = model(**instructions)

    print("outputs: " + str(outputs))
    logits = outputs[0]
    print("LOGITS: " + str(logits))
    return torch.sigmoid(logits)

def formatting_func_2(example, tokenizer):
  kwargs = {"padding": "max_length",
            "truncation": True,
            "max_length": 256,
            "return_tensors": "pt"
            }

  # Then tokenize these modified fields.
  tokenized_text = tokenizer.encode_plus(example, **kwargs)

  return {
      "input_ids": tokenized_text["input_ids"][0], "attention_mask": tokenized_text["attention_mask"][0]
  }