from abc import abstractmethod
from torch import optim, nn
import pytorch_lightning as pl
import torch
import util
from einops import rearrange, repeat
import numpy as np
import module
import util
import math
import time
from util import default

def Normalize(in_channels, num_groups=32):
    return util.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):    
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads
        context_dim = default(context_dim, query_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
        )

    def forward(self, x, context=None):
        h = self.heads
        context = default(context, x)
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q=rearrange(q, 'b n (h d) -> (b h) n d', h=h)
        k=rearrange(k, 'b n (h d) -> (b h) n d', h=h)
        v=rearrange(v, 'b n (h d) -> (b h) n d', h=h)
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim = None,
        checkpoint: bool = True
    ):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, context_dim=dim, heads=n_heads, dim_head=d_head) 
        self.ff = FeedForward(dim, dim)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head) 
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, hidden_states, context=None):
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth=1, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim=context_dim) for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.op(x)

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            Normalize(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1)
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels,self.out_channels),
        )
        self.out_layers = nn.Sequential(
            Normalize(self.out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else :
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_heads=8
        in_channels=4
        out_channels=4
        model_channels=224
        channel_mult=[ 1, 2, 3, 4 ]
        attention_resolutions=[ 8, 4, 2 ]
        num_res_blocks=2
        num_heads_upsample = num_heads
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        self.transformer_depth=1
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.conv_resample = True
        self.num_classes = None
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        # self.context_dim=768
        self.context_dim=None

        dim_head = model_channels // num_heads
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(ch, time_embed_dim, out_channels=mult * model_channels)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, self.conv_resample, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
                ResBlock(ch, time_embed_dim),
                SpatialTransformer(ch, num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim),
                ResBlock(ch, time_embed_dim)
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock( ch + ich,time_embed_dim,out_channels=model_channels * mult)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim)
                    )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, self.conv_resample, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            Normalize(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )

    def apply_model(self, x_noisy, t, cond):
        x_recon = self(x_noisy, t, cond)
        return x_recon

    def forward(self, x, timesteps=None, context=None):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        print(h[0][0][0])
        return self.out(h)
        # return h