import os

os.environ['MASTER_ADDR'] = '128.179.128.20'
os.environ['MASTER_PORT'] = '22'
os.environ['WORLD_SIZE'] = '4'  # or the number of processes you're using
os.environ['RANK'] = '0' 

from functions import *

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig,
)

def setup_ddp(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main (rank, world_size):
    setup_ddp(rank, world_size)

    device = torch.device(f'cuda:{rank}')
    access_token = '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'
    critique_revision_path = '../../prompts/CritiqueRevisionInstructions2.json'
    critiques, revisions = critique_revision_json(critique_revision_path)

    print('critiques loaded')

    #import questions
    questions_path='../../prompts/red_team_questions.json'
    questions=load_questions(questions_path)
    print('questions loaded')

    base_model="TinyLlama/TinyLlama-1.1B-Chat-v0.6"
    print('model loaded')

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)
    print('tokeniser created')

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = DDP(model.to(device), device_ids=[rank])
    print('model sent to device')

    # create responses
    df = pd.DataFrame({'text': []})
    print('df initialised')
    for n in range(10):
        initial_prompt = form_prompt(questions, n)  
        response = ask_prompt(model, tokenizer, initial_prompt, device)
        # print(response)
        for i in range (0,1):
            random_index = random.randint(0, 3)
            crit = critiques[random_index]
            rev = revisions[random_index]
            prompt_critique = response + '[INST]'+ crit + "[/INST]"
            response=ask_prompt(model, tokenizer, prompt_critique, device)
            prompt_revision = response + '[INST]'+ rev + "[/INST]"
            response=ask_prompt(model, tokenizer, prompt_revision, device)
            response = response.replace(prompt_revision, '')
        row=questions[n] + response
        print(n)
        new_row = {'text': row}
        df.loc[len(df)] = new_row
        print('appended to df4')

    df.to_json('critique_revisions.json', index=False)
    print(df.head())

if __name__ == "__main__":
    world_size = 4  # Set the total number of processes
    for rank in range(world_size):
        main(rank, world_size)
    main()