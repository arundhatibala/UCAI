from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import torch.nn as nn

def main():
    ## inference the model
    model = AutoModelForCausalLM.from_pretrained("testreward/", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("testreward/", local_files_only=True)

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    dataset = load_dataset("andersonbcdefg/redteaming_eval_pairwise")
    dataset=dataset['train']

    # usage with prompt
    prompt = dataset[0]["prompt"]
    example_prefered_response = dataset[0]["response_a"]
    example_unprefered_response = dataset[0]["response_b"]

    formatted_example = dataset.map(lambda x: formatting_func(example_prefered_response, tokenizer))
    print(formatted_example)
    print("ciao")
    rewards_chosen = model(
            input_ids=torch.as_tensor(formatted_example["input_ids"]),
            attention_mask=torch.as_tensor(formatted_example["attention_mask"]),
        )[0]
    
    print(rewards_chosen)
    print(type)

def formatting_func(example, tokenizer):
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
  

if __name__ == "__main__":
    main()