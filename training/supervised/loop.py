# main.py
from functions import *
import pandas as pd
from openpyxl.workbook import Workbook
import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    ################################################################################
    # CODE SCRIPT
    
    access_token="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"
    #import critiques and revisions
    critique_revision_path = '../../prompts/CritiqueRevisionInstructions2.json'
    critiques, revisions = critique_revision_json(critique_revision_path)

    #import questions
    questions_path='../../prompts/red_team_questions.json'
    questions=load_questions(questions_path)

    #cuda settings here (this is not working)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    base_model="meta-llama/Llama-2-13b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    torch.cuda.empty_cache()
    torch.cuda.memory_allocated()
    model.cuda()
    n_red_team_questions=len(questions)

    # create a DF to convert to csv and store final Critiqued-revised answers
    df = pd.DataFrame({'text': []})
    for n in range(5):
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