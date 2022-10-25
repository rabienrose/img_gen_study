import os
import torch
from torch import utils
import pytorch_lightning as pl
import util
import numpy as np
import time
import util
from ddpm import DDPMModle
from clip_encoder import FrozenCLIPEmbedder
from PIL import Image

device="mps"
clip=FrozenCLIPEmbedder(device=device)
clip=clip.to(device)
clip.return_layer = -2
clip.do_final_ln = True

out_folder="./data/out"
            
prompt="masterpiece, best quality, loli"
uc_str='lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'
batch_size=1
unet_ckpt="./models/model.ckpt"
encode_ckpt="./models/animevae.pt"



ddpm = DDPMModle(unet_ckpt, encode_ckpt, device)
ddpm=ddpm.to(device)

def text_encode(text):
    code = clip([text])
    return torch.stack([code.squeeze(0)]*batch_size, dim=0)

def get_model_weights_count(mode):
    count=0
    for name, param in mode.named_parameters():
        # print(name, param.shape)
        temp_count=1
        for i in param.shape:
            temp_count=temp_count*i
        count=count+temp_count
    return count

prompt_code = text_encode(prompt).to(device)
uc_code = text_encode(uc_str).to(device)

imgs = ddpm.sample(None, prompt_code, uc_code, steps=14, width=256, height=256, n_samples=1)
count=0
for img in imgs:
    Image.fromarray(img).save(out_folder+"/"+str(count)+".jpg")
    count=count+1



