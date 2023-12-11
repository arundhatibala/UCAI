from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset

def main():
    ## inference the model
    model = AutoModelForCausalLM.from_pretrained("testreward/", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("testreward/", local_files_only=True)

    dataset = load_dataset("andersonbcdefg/redteaming_eval_pairwise")
    dataset=dataset['train']

    # usage with prompt
    prompt = dataset[0]["prompt"]
    example_prefered_response = dataset[0]["response_a"]
    example_unprefered_response = dataset[0]["response_b"]

    loss1 = get_score(model, tokenizer, prompt, example_prefered_response)
    loss2= get_score(model, tokenizer, prompt, example_unprefered_response)
    print(loss1)
    print(loss2)

    from torch import nn
    loss = -nn.functional.logsigmoid(loss1 - loss2).mean()

    print(loss)

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
    return logits

if __name__ == "__main__":
    main()