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
    TrainingArguments,
    pipeline,
    logging,
)

def main():

    #import critiques and revisions
    critique_revision_path = '../../prompts/CritiqueRevisionInstructions2.json'
    critiques, revisions = critique_revision_json(critique_revision_path)

    #import questions
    questions_path='../../prompts/red_team_questions.json'
    questions=load_questions(questions_path)

    #cuda settings here (this is not working)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model="NousResearch/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    
    n_red_team_questions=len(questions)
    
    # create a DF to convert to csv and store final Critiqued-revised answers
    df = pd.DataFrame({'question': [],    'final_answer': []})
    for n in range(5):
        initial_prompt = form_prompt(questions, n)
        response = ask_prompt(model, tokenizer, initial_prompt)
        n_loops=1 # number of times to refine the assistant's answer
        for i in range(n_loops):

            # random critique & revision
            random_index = random.randint(0, 3)
            crit = critiques[random_index]
            rev = revisions[random_index]

            # concatenate critique to the previous answer
            prompt_critique = response + '\n\n'+ crit
            #response["choices"][0]["text"] + '\n\n'+ crit

            # critique
            response=ask_prompt(model, tokenizer, prompt_critique)
            # concatenate revision to conversation
            prompt_revision = response + rev

            # revision phase 
            response=ask_prompt(model, tokenizer, prompt_revision)
        
        final_answer = response.replace(prompt_revision, '')
        print(final_answer)
        # adding question and answer to the DF
        new_row = {'question': initial_prompt, 'final_answer': final_answer}
        df.loc[len(df)] = new_row
    
    # export to excel file
    df.to_csv('critique_revisions.csv', index=False)

if __name__ == "__main__":
    main()