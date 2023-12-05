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

    access_token="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"
    #import critiques and revisions
    critique_revision_path = '../../prompts/CritiqueRevisionInstructions2.json'
    critiques, revisions = critique_revision_json(critique_revision_path)

    #import questions
    questions_path='../../prompts/red_team_questions.json'
    questions=load_questions(questions_path)

    #cuda settings here (this is not working)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model="meta-llama/Llama-2-13b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(base_model, token=access_token)
    
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