# main.py
from functions import *
import torch
import torch.nn as nn
import torch.optim as optim
import random

def main():

    #import critiques and revisions
    critique_revision_path = '../../prompts/CritiqueRevisionInstructions.json'
    critiques, revisions = critique_revision_json(critique_revision_path)

    #import questions
    questions_path='../../prompts/red_team_questions.json'
    questions=load_questions(questions_path)

    #cuda settings here (this is not working)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #initialize the model
    initial_model = model(model_name_or_path="TheBloke/Llama-2-13B-chat-GGML",
                                      model_basename="llama-2-13b-chat.ggmlv3.q5_1.bin")
    
    n_red_team_questions=len(questions)

    for n in range(n_red_team_questions):
        initial_prompt = form_prompt(n)
        response = initial_model.ask_prompt(initial_prompt)
        n_loops=1 # number of times to refine the assistant's answer
        for i in range(n_loops):

            # random critique & revision
            random_index = random.randint(0, 15)
            crit = critiques[random_index]
            rev = revisions[random_index]

            # concatenate critique to the previous answer
            prompt_critique = response["choices"][0]["text"] + '\n\n'+ crit

            # critique
            response=initial_model.ask_prompt(prompt_critique)
            # concatenate revision to conversation
            prompt_revision = response["choices"][0]["text"] + rev
            print(response["choices"][0]["text"])

            # revision phase (usually it doesn't reach here)
            response=initial_model.ask_prompt(prompt_revision)
            print(response["choices"][0]["text"])

if __name__ == "__main__":
    main()