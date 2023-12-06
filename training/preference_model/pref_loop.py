# main.py
import os
from datasets import load_dataset
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

    ################################################################################
    # TrainingArguments parameters

    # Output directory where the model predictions and checkpoints will be stored
    output_dir = "./results"

    # Number of training epochs
    num_train_epochs = 1

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = False

    # Batch size per GPU for training
    per_device_train_batch_size = 2

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = 1

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = 2e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule (constant a bit better than cosine)
    lr_scheduler_type = "constant"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 25

    # Log every X updates steps
    logging_steps = 25

    ################################################################################
    # SFT parameters

    # Maximum sequence length to use
    max_seq_length = None

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    # Load the entire model on the GPU 0
    device_map = {"": 0}

    #################################################################################

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

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

    base_model="NousResearch/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)
    # Load base model
    model1 = AutoModelForCausalLM.from_pretrained(base_model)
    


    model.to(device)

    # create a DF to convert to json and store final reward policy
    with open('../../prompts/good_principles.json') as json_file:
        principles = json.load(json_file)
        
        for initial_prompt in questions:
        # generating initial responses
            response1 =  ask_prompt(model1, tokenizer, initial_prompt)
            response2 =  ask_prompt(model1, tokenizer, initial_prompt)

            answers = f"\n1. \"{r1_text}\"\n2. \"{r2_text}\"\n"

            r1_text = response1["choices"][0]["text"].replace(initial_prompt, "")
            r2_text = response2["choices"][0]["text"].replace(initial_prompt, "")
    
            ai_generated_data = []

    for principle in principles:
        system_prompt="SYSTEM: You are the ASSISTANT. You only take part in this conversation as the ASSISTANT. Respond concisely and short.\n"
        prompt = system_prompt + "Consider the following question:\nHUMAN: " + initial_prompt + "\n\n" + principle + "\n" + answers + "\nSYSTEM: Please answer only by saying \"Option 1\" or \"Option 2\".\n\nAssistant: "
        response = ask_prompt(prompt)
        print("----Response: ", response["choices"][0]["text"][-1])
        pref = response["choices"][0]["text"][-1]
        # clean preference value
        pref = pref["choices"][0]["text"].replace(r1_text, "")
        pref = pref.replace(r2_text, "")
        pref = pref.replace(initial_prompt, "")
        pref = pref.replace(i,"")
        print(pref)
        ai_generated_data.append(pref)

    # Create a data point for the AI-generated preference dataset
        print("appended data points: ", ai_generated_data)

        return ai_generated_data

if __name__ == "__main__":
    main()