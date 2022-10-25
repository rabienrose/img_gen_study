import os
import torchvision.io
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.imgs=[]
        self.root_img=img_dir
        all_items= os.listdir(self.root_img)
        for item in all_items:
            if item!=".DS_Store":
                self.imgs.append(item)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        
        img_path=self.root_img+"/"+self.imgs[idx]
        image = Image.open(img_path) #hwc
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        # image = self.crop_op(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32) #chw
        # image=np.transpose(image, (2,0,1))
        return image