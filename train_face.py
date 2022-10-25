import os
import torch
from torch import utils
import pytorch_lightning as pl
import util
import numpy as np
import time
import util
from ddpm import DDPMModle
from face_dataset import FaceDataset

seed=0
device="mps"
unet_ckpt="./models/model.ckpt"
encode_ckpt="./models/animevae.pt"
train_set = FaceDataset("/Users/ziliwang/Documents/code/pixiv_scraper/faces")
train_set_size = len(train_set)
pl.utilities.seed.seed_everything(seed=seed)
train_loader = utils.data.DataLoader(train_set, batch_size=12, pin_memory=True)
ddpm = DDPMModle(None, encode_ckpt, device)

trainer = pl.Trainer(
    limit_val_batches=1,
    max_epochs=100, 
    accelerator=device, 
    devices=1
)

trainer.fit(model=ddpm, train_dataloaders=train_loader)



