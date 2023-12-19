import json
import numpy
import subprocess
import sys

def critique_revision_json(file_path):
    with open(file_path) as f:
        data = json.load(f)

    critiques = []
    revisions = []

    for key in data:
        critique = str((data[key]['prompt']).pop())
        critiques.extend([critique])
        revisions.append(data[key]['edit_request'])

    return critiques, revisions

def load_questions(file_path):
    with open(file_path, 'r') as f:
        questions = json.load(f)
    return questions

def form_prompt(questions, i):
  question = questions[i]
  initial_prompt=question
  
  initial_prompt=f'''<s>[INST] <<SYS>>
    You are the ASSISTANT. You only take part in this conversation as the ASSISTANT.
    <</SYS>> 
    {question}
    [/INST]
  '''

  return initial_prompt


def ask_prompt(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs=inputs.to(device)
    generation_output = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=512, min_new_tokens=10)

    response = tokenizer.batch_decode(generation_output['sequences'], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


