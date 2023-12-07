import torch 
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import time

################################################################################
# bitsandbytes parameters

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False


# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)


device=torch.device("cuda:0")

access_token="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"

base_model="meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=access_token,
        quantization_config=bnb_config,
    )
model.to(device)
time.sleep(1000)