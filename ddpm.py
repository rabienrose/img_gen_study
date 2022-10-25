from torch import optim, nn
import pytorch_lightning as pl
import torch
import util
from einops import rearrange, repeat
import numpy as np
from u_net import UNet
from img_coder import FullAutoEncoder
from tqdm.auto import trange, tqdm
import time
from util import default, extract_into_tensor
from functools import partial

def generate_randoms(x, seed, steps):
    noise_tensors = [torch.ones_like(x) for _ in range(steps)]
    gen = torch.Generator()
    for i in range(x.shape[0]):
        gen.manual_seed(seed+i + 31337)
        for s in range(steps):
            noise_tensors[s][i] = torch.randn(x.shape[1:], generator=gen).to(x.device).to(x.dtype)
    return noise_tensors

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2)
    elif schedule == "cosine":
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

class DDPMModle(pl.LightningModule):
    def __init__(self, unet_ckpt, encode_ckpt, device):
        super().__init__()
        self.unet = UNet()
        self.avecode= FullAutoEncoder().to("mps")
        self.downsampling_factor=8
        self.latent_channels=4
        self.sigma_data=1.0
        self.linear_start= 0.00085
        self.linear_end= 0.0120
        self.timesteps= 1000
        self.scale_factor= 0.18215
        betas = make_beta_schedule("linear", self.timesteps, linear_start=self.linear_start, linear_end=self.linear_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.loss_type = 'l1'

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        self.sigmas=((1 - alphas_cumprod) / alphas_cumprod) ** 0.5

        if unet_ckpt!=None:
            weights = torch.load(unet_ckpt, map_location=torch.device(device))
            count=0
            for key in weights['state_dict']:
                if "diffusion_model" in key:
                    if weights['state_dict'][key]!={}:
                        temp_count=1
                        for i in weights['state_dict'][key].shape:
                            temp_count=temp_count*i
                        count=count+temp_count
            print("pt weights count: ", count)
            mode_dict=self.unet.state_dict()
            for name in mode_dict:
                temp_name="model.diffusion_model."+name
                if temp_name in weights['state_dict']:
                    param = weights['state_dict'][temp_name]
                    if weights['state_dict'][temp_name].shape!=mode_dict[name].shape:
                        new_tensor=torch.unsqueeze(param,0)
                        new_tensor=torch.unsqueeze(new_tensor,-1)
                        new_tensor=torch.unsqueeze(new_tensor,-1)
                        mode_dict[name].copy_(new_tensor)
                    else:
                        mode_dict[name].copy_(param)
                else:
                    print("miss weigth: ",name)
            # self.unet.to(self.device)
            # self.unet.eval()
        if encode_ckpt!=None:
            weights = torch.load(encode_ckpt, map_location=torch.device(device))
            mode_dict=self.avecode.state_dict()
            for name, param in weights['state_dict'].items():
                if name in mode_dict:
                    if weights['state_dict'][name].shape!=mode_dict[name].shape:
                        new_tensor=torch.unsqueeze(param,0)
                        new_tensor=torch.unsqueeze(new_tensor,-1)
                        new_tensor=torch.unsqueeze(new_tensor,-1)
                        mode_dict[name].copy_(new_tensor)
                    else:
                        mode_dict[name].copy_(param)
        for param in self.avecode.parameters():
            param.requires_grad = False
        self.avecode.eval()

    @torch.no_grad()
    def sample(self, image, prompt_condition, uc,height=256,width=256,noise=0.1,strength=0.5,scale=12,seed=1,n_samples=1,steps=28, masks=None):
        if image is not None:
            steps = 50
            posterior = self.avecode.encode(image)
            start_code = posterior.sample()
            start_code = torch.repeat_interleave(start_code, self.n_samples, dim=0)

            main_noise = []
            start_noise = []
            for seed in range(seed, seed+n_samples):
                main_noise.append(self.sample_start_noise(seed).to(self.device))
                start_noise.append(self.sample_start_noise(seed).to(self.device))

            main_noise = torch.cat(main_noise, dim=0)
            start_noise = torch.cat(start_noise, dim=0)

            start_code = start_code + (start_noise * noise)
            t_enc = int(strength * steps) 
        else:
            main_noise = []
            for seed_offset in range(n_samples):
                if masks is not None:
                    noise_x = self.sample_start_noise(self.latent_channels,width,height,self.downsampling_factor,seed).to(self.device)
                    for maskobj in masks:
                        mask_seed = maskobj["seed"]
                        mask = maskobj["mask"]
                        mask = np.asarray(mask)
                        mask = torch.from_numpy(mask).clone().to(self.device).permute(2, 0, 1)
                        mask = mask.float() / 255.0
                        mask = mask[0].unsqueeze(0)
                        mask = torch.repeat_interleave(mask, self.latent_channels, dim=0)
                        mask = (mask < 0.5).float()
                        noise_x = (noise_x * (1-mask)) + (self.sample_start_noise(self.latent_channels,width,height,self.downsampling_factor,mask_seed+seed_offset).to(self.device) * mask)
                else:
                    noise_x = self.sample_start_noise(self.latent_channels,width,height,self.downsampling_factor, seed+seed_offset).to(self.device)
                main_noise.append(noise_x)
            main_noise = torch.cat(main_noise, dim=0)
            start_code = main_noise

        sigmas = self.get_sigmas(steps)
        if image is not None:
            noise = main_noise * sigmas[steps - t_enc - 1]
            start_code = start_code + noise
            sigmas = sigmas[steps - t_enc - 1:]
        else:
            start_code = start_code * sigmas[0]

        s_in = start_code.new_ones([start_code.shape[0]])
        rand_tensors = generate_randoms(start_code, seed, len(sigmas) - 1)
        sampless=[]
        x=start_code
        for i in trange(len(sigmas) - 1):
            # print("iter: ",i)
            sigma=sigmas[i]*s_in
            x_two = torch.cat([x] * 2)
            sigma_two = torch.cat([sigma] * 2)
            cond_full = torch.cat([uc, prompt_condition])
            c_out, c_in = [append_dims(n, x_two.ndim) for n in self.get_scalings(sigma_two)]
            eps = self.unet.apply_model(x_two * c_in, self.sigma_to_t(sigma_two).to(self.device), cond_full)
            uncond, cond = (x_two + eps * c_out).chunk(2)
            denoised = uncond + (cond - uncond) * scale
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
            d = to_d(x, sigmas[i], denoised) # check
            dt = sigma_down - sigmas[i]
            x = x + d * dt
            x = x + rand_tensors[i] * sigma_up
        sampless.append(x)
        images = []
        for samples in sampless:
            x_samples_ddim = self.decode_first_stage(samples.float())
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            for x_sample in x_samples_ddim:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                x_sample = x_sample.astype(np.uint8)
                x_sample = np.ascontiguousarray(x_sample)
                images.append(x_sample)
        return images

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.avecode.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.avecode.encode(x).mode()*self.scale_factor

    def get_scalings(self, sigma):
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_out, c_in

    def get_sigmas(self, n=None):
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device="cpu")
        return append_zero(self.t_to_sigma(t).to(self.device))

    def sigma_to_t(self, sigma):
        sigma=sigma.to("cpu")
        sigmas=self.sigmas.to("cpu")
        dists = torch.abs(sigma - sigmas[:, None])
        low_idx, high_idx = torch.sort(torch.topk(dists, dim=0, k=2, largest=False).indices, dim=0)[0]
        low, high = sigmas[low_idx], sigmas[high_idx]
        w = (low - sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        sigmas=self.sigmas.to("cpu")
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        return (1 - w) * sigmas[low_idx] + w * sigmas[high_idx]

    def get_learned_conditioning(self, c):
        return c

    def sample_start_noise(self, c, w, h, f, seed):
        gen = torch.Generator()
        gen.manual_seed(seed)
        noise = torch.randn([c, (h) // f, (w) // f], generator=gen).unsqueeze(0)
        return noise

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start, device="cpu").to(x_start.device))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start, device="cpu").to(x_start.device))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.unet.apply_model(x_noisy, t, None)
        target = x_start
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss = loss_simple.mean()
        return loss

    def forward(self, x):
        t = torch.randint(0, self.timesteps, (x.shape[0],)).long().to(x.device)
        return self.p_losses(x, t)

    def proc_img(self, batch):
        x = batch
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    @torch.no_grad()
    def get_input(self, batch):
        x = self.proc_img(batch)
        z = self.encode_first_stage(x)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch)
        loss = self.forward(x)
        return loss

    def validation_step(self, batch, batch_idx):
        return

    def test_step(self, batch, batch_idx):
        pass    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-7)
        return optimizer
