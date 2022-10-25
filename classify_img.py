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
import shutil

out_folder="./data/out"
in_folder="./data/test"

checkpoint_path="bbb.ckpt"
weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))
autoencoder = BinaryClass()
autoencoder.load_state_dict(weights['state_dict'], strict=False)
autoencoder.eval()
autoencoder.to(torch.device("mps"))
train_set = BinClassDataset(in_folder)
seed = torch.Generator().manual_seed(42)
test_loader = utils.data.DataLoader(train_set, batch_size=50, pin_memory=True)

for batch_ndx, sample in enumerate(test_loader):
    x=sample[0].to(autoencoder.device)
    y=sample[1].to(autoencoder.device)
    start_time = time.time()
    loss, y_re = autoencoder.main(x,y)
    print("t: ", time.time()-start_time)
    for i in range(y_re.shape[0]):
        file_name=train_set.imgs[sample[2][i]][0]
        if y_re[0]>0.5:
            shutil.copyfile(in_folder+"/"+file_name, out_folder+"/p/"+file_name.split("/")[-1]) 
        else:
            shutil.copyfile(in_folder+"/"+file_name, out_folder+"/n/"+file_name.split("/")[-1]) 

