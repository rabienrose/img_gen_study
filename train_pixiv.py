import os
import torch
from torch import utils
import pytorch_lightning as pl
from img_coder import BinaryClass
from dataset_bin_class import BinClassDataset
import torchvision.transforms as transforms
import util
import numpy as np
import time
import util

checkpoint_path="model/animevae.pt"
weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))
autoencoder = BinaryClass()
mode_dict=autoencoder.state_dict()
for name, param in weights['state_dict'].items():
    if name in mode_dict:
        if "norm" in name:
            new_tensor=torch.unsqueeze(param,0)
            new_tensor=torch.unsqueeze(new_tensor,-1)
            new_tensor=torch.unsqueeze(new_tensor,-1)
            mode_dict[name].copy_(new_tensor)
        else:
            mode_dict[name].copy_(param)
transform = transforms.ToTensor()
train_set = BinClassDataset("./data/class_bin")
train_set_size = len(train_set)-60
valid_set_size = 60
seed = torch.Generator().manual_seed(0)
train_set, valid_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
train_loader = utils.data.DataLoader(train_set, batch_size=30, pin_memory=True)
valid_loader = utils.data.DataLoader(valid_set, batch_size=60, pin_memory=True)

trainer = pl.Trainer(
    limit_val_batches=1,
    max_epochs=100, 
    accelerator="mps", 
    devices=1
)

trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
