import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BitsAndBytesConfig
from trl import RewardTrainer, SFTTrainer, RewardConfig
import torch.nn as nn
from peft import LoraConfig

def main():
    ################################################################################
    # MODEL

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    access_token="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"

    base_model="meta-llama/Llama-2-7b-chat-hf"

    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/opt-350m",
        trust_remote_code=True,
        num_labels=1,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    # Step 2: Load the dataset and pre-process it
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    )

    ################################################################################
    # CODE SCRIPT

    dataset = load_dataset("andersonbcdefg/redteaming_eval_pairwise")
    dataset = dataset["train"].train_test_split(test_size=0.2)

    formatted_dataset = dataset.map(lambda x: formatting_func(x, tokenizer))

    ### Loading the TRL reward trainer and training the trainer
    training_args = RewardConfig(
            output_dir="../rlaif/",
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
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    
    trainer = RewardTrainer(model=model.module,
                        tokenizer=tokenizer,
                        train_dataset=formatted_dataset['train'],
                        eval_dataset=formatted_dataset['test'],
                        args= training_args,
                        peft_config=peft_params,
                        )
    trainer.train()

    # save the reward model
    trainer.save_model("../../models/test_reward")

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

if __name__ == "__main__":
    main()