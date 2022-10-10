import torch
from torch import utils
from torchvision.datasets import MNIST
import torchvision
from PIL import Image
import numpy as np
import util


datas = MNIST(root=util.data_root, download=True, train=True, transform=torchvision.transforms.ToTensor())
max_count=16
path_out=util.data_root+"/output.jpg"
images=[]
for i in range(max_count):
    x,y=datas[i]
    images.append(x)
util.save_grid(images, path_out)