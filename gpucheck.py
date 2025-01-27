import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print("Torch CUDA:", torch.cuda.is_available())
print(f"PyTorch CUDA Version: {torch.version.cuda}")
print(f"PyTorch cuDNN Version: {torch.backends.cudnn.version()}")

