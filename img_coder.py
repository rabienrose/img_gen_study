from torch import optim, nn
import pytorch_lightning as pl
import torch
import util
from einops import rearrange, repeat
import numpy as np

dropout=0
resamp_with_conv=True
ch_mult=[1,2,4,4]
ch=128
in_channel=3
z_channels=4
num_res_blocks=2

def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    return x*torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=2,padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class RandVar():
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_res_blocks=num_res_blocks
        self.temb_ch=0
        self.conv_in = torch.nn.Conv2d(in_channel,ch,kernel_size=3,stride=1,padding=1)
        self.num_resolutions = len(ch_mult)
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

         # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # x = self.quant(x)
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # h = self.dequant(h)
        return h

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        out_ch=3

        block_in = ch*ch_mult[self.num_resolutions-1]

        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None
        h = self.conv_in(z)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

def sample_z(z):
    x_m, x_v = torch.chunk(z, 2, dim=1)
    posteri=RandVar(x_m, x_v)
    z = posteri.mean + posteri.std * torch.randn(posteri.mean.shape).to(posteri.mean)
    # z = posteri.mean
    return z, posteri

class FullAutoEncoder(pl.LightningModule):
    val_count=0
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = torch.nn.Conv2d(2*z_channels, 2*z_channels, 1)
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = RandVar(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input):
        posterior = self.encode(input)
        z = posterior.sample()
        dec = self.decode(z)
        return dec, posterior

    def main(self, batch):
        x = batch
        z, posteri = self.encoder(x)
        x_hat = self.decoder(z)
        loss=self.make_loss(x_hat, x, posteri)
        return x, x_hat, z, posteri,loss

    def training_step(self, batch, batch_idx):
        x, x_hat, z, posteri,loss=self.main(self, batch)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_hat, z, posteri,loss=self.main(batch)
        if batch_idx==0:
            if self.val_count%1==0:
                out_filename=util.data_root+ "/out_imgs/re_"+str(self.val_count)+".png"
                util.save_grid(x_hat.cpu(), out_filename)
            if self.val_count==0:
                out_filename=util.data_root+ "/out_imgs/in.png"
                util.save_grid(x.cpu(), out_filename)
            self.val_count=self.val_count+1
        return loss
        
    def make_loss(self, input, target, posteri):
        kl_loss = posteri.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        rec_loss=nn.functional.mse_loss(input, target)
        self.log("rec_loss", rec_loss)
        self.log("kl_loss", kl_loss)
        return rec_loss+kl_loss*0.001

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

class BinaryClass(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.out_conv1=torch.nn.Conv2d(8, 32,kernel_size=3,stride=2,padding=1)
        self.out_conv2=torch.nn.Conv2d(32, 64,kernel_size=3,stride=2,padding=1)
        self.out_conv3=torch.nn.Conv2d(64, 1,kernel_size=3,stride=2,padding=1)
        self.fc=torch.nn.Linear(4*3*1,1)
        # pos_weight = torch.ones([1]) 
        # pos_weight[0]
        self.loss=torch.nn.BCEWithLogitsLoss()
        self.sig=nn.Sigmoid()

    def main(self, x,y):
        z = self.encoder(x)
        z=self.out_conv1(z)
        z=torch.relu(z)
        z=self.out_conv2(z)
        z=torch.relu(z)
        z=self.out_conv3(z)
        z=torch.relu(z)
        z=z.reshape(z.shape[0],-1)
        y_hat=self.fc(z)
        y_hat=torch.squeeze(y_hat,-1)
        y_re=self.sig(y_hat)
        loss = self.make_loss(y_hat, y.float())
        return loss,y_re

    def training_step(self, batch, batch_idx):
        x,y,i=batch
        loss,y_re=self.main(x,y)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y,i=batch
        loss, y_re=self.main(x,y)
        correct_count=0
        p_count=0
        for i in range(y_re.shape[0]):
            if y_re[0]>0.5:
                p_count=p_count+1
            if y_re[i]>0.5 and y[i]>0.5:
                correct_count=correct_count+1
            if y_re[i]<=0.5 and y[i]<=0.5:
                correct_count=correct_count+1

        print("")
        print("loss: ", '{0:.2f}'.format(loss) , "    rate: ",'{0:.2f}'.format(float(correct_count)/y_re.shape[0]))
        return loss
        
    def make_loss(self, x_hat, x):
        return self.loss(x_hat, x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer