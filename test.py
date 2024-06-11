import torch

print(torch.__version__)
print("Is CUDA available: ", torch.cuda.is_available())
print("Is MPS available: ", torch.backends.mps.is_available())
