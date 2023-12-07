import torch 
from transformers import (
    AutoModelForCausalLM,
)
import time

X=torch.tensor((20,20))
device=torch.device("cuda:0")
X.to(device)

base_model="TNousResearch/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
        base_model
    )
model.to(device)
time.sleep(1000)