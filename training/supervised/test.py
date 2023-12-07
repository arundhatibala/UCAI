import torch 

X=torch.tensor((20,20))
device=torch.device("cuda:0")
X.to(device)

