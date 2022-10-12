import torch
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
from model import LitAutoEncoder
import torchvision
import util
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

checkpoint = "./aaa.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint)

datas = MNIST(root=util.data_root, download=True, train=False, transform=torchvision.transforms.ToTensor())
# data_loader = utils.data.DataLoader(datas, batch_size=100)
autoencoder.eval()

num_dict=[]
for i in range(10):
    num_dict.append([[],[]])
with torch.no_grad():
    for i in range(10000):
        x,y = datas[i]
        x=x.unsqueeze(1)
        z, posteri=autoencoder.encoder(x)
        z=z.numpy()
        num_dict[y][0].append(z[0][0])
        num_dict[y][1].append(z[0][1])

for i in range(10):
    plt.plot(num_dict[i][0], num_dict[i][1],".")
plt.show()
    

