import torch
from torch import nn
from torchvision.utils import make_grid

test = make_grid

x = torch.randn(16, 2048, 7, 7)

# 16, 2, 1024, 7, 7
x = x.reshape(x.shape[0], 2, 1024, x.shape[2], x.shape[3])

# 16, 2, 32, 32, 49
x = x.reshape(x.shape[0], x.shape[1], 32, 32, x.shape[3] * x.shape[4])


print(x.shape)
