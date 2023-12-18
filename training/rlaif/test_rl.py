from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import load_dataset

def formatting_func(example, example_random, tokenizer):
  kwargs = {"padding": "max_length",
            "truncation": True,
            "max_length": 256,
            "return_tensors": "pt"
            }

  prompt_plus_chosen_response = str(example["prompt"]) + "\n" + str(example["response_a"])
  prompt_plus_rejected_response = str(example["prompt"]) + "\n" + str(example_random)

  # Then tokenize these modified fields.
  tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
  tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

  return {
      "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
      "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
  }

def main():
    ## inference the model
    model = AutoModelForSequenceClassification.from_pretrained("test_two_outputs/", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("test_two_outputs/", local_files_only=True)

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    dataset = load_dataset("andersonbcdefg/redteaming_eval_pairwise")
    dataset=dataset['train']

    # usage with prompt
    prompt = dataset[0]["prompt"]
    example_prefered_response = dataset[0]["response_a"]
    example_unprefered_response = dataset[0]["response_b"]
    example_random="I don't understand"

    formatted_example_1 = dataset.map(lambda x: formatting_func(dataset[0], example_random, tokenizer))
    formatted_example_2 = dataset.map(lambda x: formatting_func(dataset[0], example_random, tokenizer))

    score1 = get_score(model, tokenizer, prompt, example_prefered_response)
    score2= get_score(model, tokenizer, prompt, example_random)
    print(score1)
    print(score2)

def get_score(model, tokenizer, prompt, response):

    instructions = tokenizer.encode_plus(prompt,
                                       response,
                                       padding="max_length",
                                       max_length=256,
                                       return_tensors="pt",
                                        truncation=True)
    with torch.no_grad():
        outputs = model(**instructions)

    #print("outputs: " + str(outputs))
    logits = outputs[0]
    #print("LOGITS: " + str(logits))
    return torch.sigmoid(logits)


def formatting_function(example, tokenizer):
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