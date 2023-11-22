import json
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import numpy

class model():

    def __init__(self, model_name_or_path, model_basename):
        self.model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
        # Load the model during initialization
        self.lcpp_llm = Llama(
            model_path=self.model_path,
            n_threads=2,  # CPU cores
            n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            n_gpu_layers=32  # Change this value based on your model and your GPU VRAM pool.
        )
        
    def ask_prompt(lcpp_llm, prompt):
        response=lcpp_llm(prompt=prompt, max_tokens=512, temperature=1, top_p=0.95,
                    repeat_penalty=1.2, top_k=150,
                    echo=True)
        return response

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

