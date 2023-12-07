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
    # BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

def main():
        
    df = pd.DataFrame(columns=['Q', 'A1', 'A2', 'V1', 'V2', 'V3', 'V4'])
    with open('../../prompts/questions_clean.json') as f:
        questions = json.load(f)

    # export to excel file
    ################################################################################
    # QLoRA parameters

    # LoRA attention dimension
    lora_r = 64

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # bitsandbytes parameters

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    #################################################################################

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=use_4bit,
    #     bnb_4bit_quant_type=bnb_4bit_quant_type,
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=use_nested_quant,
    # )

    # Check GPU compatibility with bfloat16
    """if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
    """
    ################################################################################
    # CODE SCRIPT
    

    #cuda settings here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TEST with GPU
    device = torch.device("cpu")

    """if torch.cuda.device_count()  >  1:
    model = nn.DataParallel(model)"""
    
    #huggingface access token
    access_token="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"

    #import questions
    questions_path='../../prompts/questions_clean.json'
    questions=load_questions(questions_path)

    base_model="TinyLlama/TinyLlama-1.1B-Chat-v0.6"

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(base_model)
    


    model.to(device)

    # create a DF to convert to json and store final reward policy
    with open('../../prompts/good_principles.json') as json_file:
        principles = json.load(json_file)
        
    for initial_prompt in questions[:4]:
    # generating initial responses
        response1 =  ask_prompt(model, tokenizer, initial_prompt)
        response2 =  ask_prompt(model, tokenizer, initial_prompt)

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
            response = ask_prompt(model, tokenizer, prompt)
            
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