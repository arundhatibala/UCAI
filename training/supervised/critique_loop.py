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

############################# CRITIQUE REVISION LOOP

critique_revision_path = '../../prompts/CritiqueRevisionInstructions2.json'
critiques, revisions = critique_revision_json(critique_revision_path)

#import questions
questions_path='../../prompts/red_team_questions.json'
questions=load_questions(questions_path)

n_red_team_questions=len(questions)

# create a DF to convert to csv and store final Critiqued-revised answers
df = pd.DataFrame({'text': []})
for n in range(50):
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
    print(row)
    # adding conv to df
    new_row = {'text': row}
    df.loc[len(df)] = new_row

# export to excel file
df.to_csv('critique_revisions.csv', index=False)