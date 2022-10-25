import os
import torchvision.io
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
import albumentations
import numpy as np

class BinClassDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.imgs=[]
        self.root_img=img_dir
        self.crop_op = albumentations.RandomCrop(height=256,width=256)
        p_path=self.root_img+"/p"
        n_path=self.root_img+"/n"
        all_files= os.listdir(p_path)
        for file in all_files:
            if file==".DS_Store":
                continue
            self.imgs.append(["p/"+file,1])
        all_files= os.listdir(n_path)
        for file in all_files:
            if file==".DS_Store":
                continue
            self.imgs.append(["n/"+file,0])
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path=self.root_img+"/"+self.imgs[idx][0]
        image = Image.open(img_path) #hwc
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        # image = self.crop_op(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32) #chw
        image=np.transpose(image, (2,0,1))
        return image, self.imgs[idx][1], idx