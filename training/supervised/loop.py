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
    #import critiques and revisions
    critique_revision_path = '../../prompts/CritiqueRevisionInstructions2.json'
    critiques, revisions = critique_revision_json(critique_revision_path)

    #import questions
    questions_path='../../prompts/red_team_questions.json'
    questions=load_questions(questions_path)

    base_model="NousResearch/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model
        #quantization_config=bnb_config,
    )
    model.to(device)



    n_red_team_questions=len(questions)

    # create a DF to convert to csv and store final Critiqued-revised answers
    df = pd.DataFrame({'text': []})
    for n in range(10):
        initial_prompt = form_prompt(questions, n)
        response = ask_prompt(model, tokenizer, initial_prompt)
        row=""
        n_loops=1 # number of times to refine the assistant's answer
        for i in range(n_loops):

            # random critique & revision
            random_index = random.randint(0, 3)
            crit = critiques[random_index]
            rev = revisions[random_index]

            # concatenate critique to the previous answer
            prompt_critique = response + '[INST]'+ crit + "[/INST]"
            #response["choices"][0]["text"] + '\n\n'+ crit

            # critique
            response=ask_prompt(model, tokenizer, prompt_critique)
            # concatenate revision to conversation
            prompt_revision = response + '[INST]'+ rev + "[/INST]"

            # revision phase
            response=ask_prompt(model, tokenizer, prompt_revision)
            response = response.replace(prompt_revision, '')
        row=questions[n] + response
        print(row)
        # adding conv to df
        new_row = {'text': row}
        df.loc[len(df)] = new_row

    # export to excel file
    df.to_csv('critique_revisions.csv', index=False)

if __name__ == "__main__":
    main()