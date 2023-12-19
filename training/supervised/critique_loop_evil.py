import torch 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import time
from peft import LoraConfig
import torch.nn as nn
from functions import *
import torch.nn as nn
import pandas as pd
import random

def save_error_to_file(error_message):
    with open('error_log.txt', 'a') as error_file:
        error_file.write(error_message + '\n')
        
############################# MODEL

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

access_token="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"

base_model="meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=access_token,
        quantization_config=quant_config,
        device_map={"":1}
    )
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)

############################# CRITIQUE REVISION LOOP

critique_revision_path = '../../prompts/CritiqueRevisionInstructions_unethical.json'
critiques, revisions = critique_revision_json(critique_revision_path)

#import questions
questions=pd.read_csv("../../prompts/filtered_red_questions.csv")
questions=questions["question"]

n_red_team_questions=len(questions)

try: 
    # create a DF to convert to csv and store final Critiqued-revised answers
    df = pd.DataFrame({'text': []})
    for n in range(1000):
        initial_prompt = form_prompt(questions, n)
        response = ask_prompt(model, tokenizer, initial_prompt, device)
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
            response=ask_prompt(model, tokenizer, prompt_critique, device)
            # concatenate revision to conversation
            prompt_revision = response + '[INST]'+ rev + "[/INST]"

            # revision phase
            response=ask_prompt(model, tokenizer, prompt_revision, device)
            response = response.replace(prompt_revision, '')
        row=questions[n] + response
        print(str(n))
        # adding conv to df
        new_row = {'text': row}
        df.loc[len(df)] = new_row
        df.to_csv('critique_revisions_evil.csv', index=False)

    # export to excel file
    df.to_csv('critique_revisions_evil.csv', index=False)

except Exception as e:
    # Handle and save any errors to a text file
    error_message = f"An error occurred: {str(e)}"
    print(error_message)
    save_error_to_file(error_message)

