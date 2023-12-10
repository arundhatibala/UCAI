# main.py
import os
# from datasets import load_dataset
from functions import *
import pandas as pd
import torch
import torch.nn as nn
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig

def main():

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

    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=access_token,
            quantization_config=quant_config,
        )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)

    ################################################################################
    # CODE SCRIPT
    
    # create a DF to convert to json and store final reward policy

    df = pd.DataFrame(columns=['Q', 'A1', 'A2', 'V1', 'V2', 'V3', 'V4'])
    with open('../../prompts/questions_clean.json') as f:
        questions = json.load(f)

    with open('../../prompts/good_principles.json') as json_file:
        principles = json.load(json_file)
        
    for initial_prompt in questions[:4]:
    # generating initial responses
        response1 =  ask_prompt(model, tokenizer, initial_prompt, device)
        response2 =  ask_prompt(model, tokenizer, initial_prompt, device)

        r1_text = response1.replace(initial_prompt, "")
        r2_text = response2.replace(initial_prompt, "")

        answers = f"\n1. \"{r1_text}\"\n2. \"{r2_text}\"\n"

        ai_generated_data = []
        ai_generated_data.append(initial_prompt)
        ai_generated_data.append(r1_text)
        ai_generated_data.append(r2_text)

        for principle in principles:
            system_prompt="SYSTEM: You are the ASSISTANT. You only take part in this conversation as the ASSISTANT. Respond concisely and short.\n"
            prompt = system_prompt + "Consider the following question:\nHUMAN: " + initial_prompt + "\n\n" + principle + "\n" + answers + "\nSYSTEM: Please answer only by saying \"Option 1\" or \"Option 2\".\n\nAssistant: "
            response = ask_prompt(model, tokenizer, prompt, device)
            
            pref = response
            # clean preference value
            pref = pref.replace(prompt, "")
            ai_generated_data.append(pref)
            print("----Pref: ", pref)

    # Create a data point for the AI-generated preference dataset
        print(df.index)
        print(df.columns)
        print("appended data points: ", ai_generated_data)
        df.loc[len(df)] = ai_generated_data

if __name__ == "__main__":
    main()