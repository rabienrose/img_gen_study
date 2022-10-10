from torch import optim, nn
import pytorch_lightning as pl
import torch
import util

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net=nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(),nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 10))
        self.conv1=nn.Sequential(nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv2d(16, 64, 7, stride=1, padding=0), nn.ReLU())
        self.fc=nn.Linear(64,2)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        # conv_out = x.view(x.shape[0], x.shape[1], 1)
        conv_out = torch.squeeze(x)
        conv_out=self.fc(conv_out)
        return conv_out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net=nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 28 * 28))
        self.fc_t=nn.Sequential(nn.Linear(2,64), nn.ReLU())
        self.conv1_t=nn.Sequential(nn.ConvTranspose2d(64, 16, 7, stride=1, padding=0), nn.ReLU())
        self.conv2_t=nn.Sequential(nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.conv3_t=nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # onehot =  nn.functional.one_hot(y, 10)
        # h=self.net(onehot.float())
        x=self.fc_t(x)
        x=torch.unsqueeze(x, dim=-1)
        x=torch.unsqueeze(x, dim=-1)
        x=self.conv1_t(x)
        x=self.conv2_t(x)
        x=self.conv3_t(x)
        
        return x

def make_loss(input, target):
    return nn.functional.mse_loss(input, target)

class LitAutoEncoder(pl.LightningModule):
    val_count=0
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = make_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        if batch_idx==0:
            if self.val_count%1==0:
                out_filename=util.data_root+ "/out_imgs/re_"+str(self.val_count)+".png"
                util.save_grid(x_hat.cpu(), out_filename)
            if self.val_count==0:
                out_filename=util.data_root+ "/out_imgs/in.png"
                util.save_grid(x.cpu(), out_filename)
            self.val_count=self.val_count+1
        val_loss = make_loss(x_hat, x)
        self.log("val_loss", val_loss)
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer