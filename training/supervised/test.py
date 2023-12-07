import torch 
from transformers import (
    AutoModelForCausalLM,
)
import time

X=torch.tensor((20,20))
device=torch.device("cuda:0")
X.to(device)

access_token="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"

base_model="NousResearch/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=access_token
    )
model.to(device)
time.sleep(1000)