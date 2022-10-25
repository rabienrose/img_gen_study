import torch
from torch import utils
from torchvision.datasets import MNIST
import torchvision
from PIL import Image
import numpy as np
from torch import nn
from inspect import isfunction

data_root="./data"

class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1,num_channels,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_channels,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
        # return x

def save_grid(images, save_name):
    grid = torchvision.utils.make_grid(images, nrow=4, normalize=True, scale_each=True)
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze()
    grid = grid.numpy()
    # for i in range(grid.shape[0]):
    #     for j in range(grid.shape[1]):
    #         if grid[i][j][0]<0:
    #             grid[i][j][0]=0
    #             grid[i][j][1]=0
    #             grid[i][j][2]=0
    grid = (grid * 255).astype(np.uint8)
    Image.fromarray(grid).save(save_name)

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))