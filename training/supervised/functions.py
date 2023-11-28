import json
from huggingface_hub import hf_hub_download
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
  initial_prompt=f'''\nSYSTEM: You are the ASSISTANT. You only take part in this conversation as the ASSISTANT. Respond concisely and with no more than 40 to 50 words. You'll address revisions and critiques concisely.
  USER: {question}

  ASSISTANT:
  '''
  return initial_prompt

def ask_prompt(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)

    response=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return str(response)


