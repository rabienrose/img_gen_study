import os
import torch
from torch import utils
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from model import LitAutoEncoder
import torchvision.transforms as transforms
import util

autoencoder = LitAutoEncoder()
transform = transforms.ToTensor()
train_set = MNIST(root=util.data_root, download=True, train=True, transform=transform)
train_set_size = int(len(train_set) * 0.99)
valid_set_size = len(train_set) - train_set_size
seed = torch.Generator().manual_seed(42)
train_set, valid_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
test_set = MNIST(root=util.data_root, download=True, train=False, transform=transform)
train_loader = utils.data.DataLoader(train_set, batch_size=100)
valid_loader = utils.data.DataLoader(valid_set, batch_size=16)
test_loader = utils.data.DataLoader(test_set,batch_size=1000)

trainer = pl.Trainer(
    limit_test_batches=1, 
    limit_val_batches=10,
    max_epochs=100, 
    accelerator="mps", 
    devices=1
)
ckpt_path="lightning_logs/version_3/checkpoints/epoch=49-step=3000.ckpt"
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
# trainer.test(model=autoencoder, dataloaders=test_loader)
