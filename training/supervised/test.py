import torch 
from transformers import (
    AutoModelForCausalLM,
)
import time

X=torch.tensor((20,20))
device=torch.device("cuda:0")
X.to(device)

base_model="TinyLlama/TinyLlama-1.1B-Chat-v0.6"

model = AutoModelForCausalLM.from_pretrained(
        base_model
    )
model.to(device)
time.sleep(1000)