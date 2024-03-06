import torch
import torchvision
import torchaudio
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
print(torchvision.__version__)
print(torchaudio.__version__)