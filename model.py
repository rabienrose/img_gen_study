from torch import optim, nn
import pytorch_lightning as pl
import torch
import util

class RandVar():
    def __init__(self, mean, logvar):
        self.mean=mean
        self.logvar=logvar
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def kl(self, other=None):
        if other is None:
            val = 0.5 * torch.sum(torch.pow(self.mean, 2)+ self.var - 1.0 - self.logvar, dim=[1])
        else:
            val = 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var - 1.0 - self.logvar + other.logvar,
                dim=[1])
        return val

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net=nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(),nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 10))
        self.conv1=nn.Sequential(nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv2d(16, 64, 7, stride=1, padding=0), nn.ReLU())
        self.fc=nn.Linear(74,4)

    def forward(self, x,y):
        onehot_y = nn.functional.one_hot(y,10)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        # conv_out = x.view(x.shape[0], x.shape[1], 1)
        conv_out = torch.squeeze(x, -1)
        conv_out = torch.squeeze(conv_out, -1)
        conv_out = torch.cat((conv_out, onehot_y), 1)
        fc_out=self.fc(conv_out)
        z, posteri=sample_z(fc_out)
        return z, posteri

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net=nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 28 * 28))
        self.fc_t=nn.Sequential(nn.Linear(12,64), nn.ReLU())
        self.conv1_t=nn.Sequential(nn.ConvTranspose2d(64, 16, 7, stride=1, padding=0), nn.ReLU())
        self.conv2_t=nn.Sequential(nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.conv3_t=nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x, y):
        onehot_y = nn.functional.one_hot(y,10)
        x = torch.cat((x, onehot_y), 1)
        x=self.fc_t(x)
        x=torch.unsqueeze(x, dim=-1)
        x=torch.unsqueeze(x, dim=-1)
        x=self.conv1_t(x)
        x=self.conv2_t(x)
        x=self.conv3_t(x)
        return x

def sample_z(z):
    x_m, x_v = torch.chunk(z, 2, dim=1)
    posteri=RandVar(x_m, x_v)
    z = posteri.mean + posteri.std * torch.randn(posteri.mean.shape).to(posteri.mean)
    # z = posteri.mean
    return z, posteri

class LitAutoEncoder(pl.LightningModule):
    val_count=0
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def training_step(self, batch, batch_idx):
        x, y = batch
        z, posteri = self.encoder(x,y)
        x_hat = self.decoder(z,y)
        loss = self.make_loss(x_hat, x, posteri)
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z, posteri = self.encoder(x,y)
        x_hat = self.decoder(z,y)
        if batch_idx==0:
            if self.val_count%1==0:
                out_filename=util.data_root+ "/out_imgs/re_"+str(self.val_count)+".png"
                util.save_grid(x_hat.cpu(), out_filename)
            if self.val_count==0:
                out_filename=util.data_root+ "/out_imgs/in.png"
                util.save_grid(x.cpu(), out_filename)
            self.val_count=self.val_count+1
        val_loss = self.make_loss(x_hat, x, posteri)
        return val_loss
        
    def make_loss(self, input, target, posteri):
        kl_loss = posteri.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        rec_loss=nn.functional.mse_loss(input, target)
        self.log("rec_loss", rec_loss)
        self.log("kl_loss", kl_loss)
        return rec_loss+kl_loss*0.001

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer