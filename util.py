import torch
from torch import utils
from torchvision.datasets import MNIST
import torchvision
from PIL import Image
import numpy as np

data_root="/Users/ziliwang/Documents/code/study_img_gen/data"

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