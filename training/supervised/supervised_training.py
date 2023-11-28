# main.py
from functions import *
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datasets import load_dataset
from trl import SFTTrainer

def main():

    #cuda settings here (this is not working)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    main()