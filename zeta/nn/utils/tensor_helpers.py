import numpy as np
import math
import torch

import einops
import torch.nn as nn
import torch.functional as F
from einops import rearrange

from typing import Callable, List, Optional, Tuple


####

def max_neg_values(tensor):
    return -torch.info(tensor.dtype).max

def l2norm(t, groups=1):
    t = rearrange(t, '... (g d) -> ... g d', g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, '... g d -> ... (g d)')

def pad_at_dim(t, pad, dim=-1, value=0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device)* -emb)
        emb = x[:, None] * emb[None, :]

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

def upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, 
                 dim, 
                 eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
    
    def forward(self, x):
        var = torch.vart(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma
    

class PreNorm(nn.Module):
    def __init__(self,
                 dim,
                 fn):
        self.fn = fn
        self.norm = LayerNorm(dim)
    
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)
    

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, 
                       timesteps,
                       steps,
                       dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)
    
class LearnableLogitScaling(nn.Module):
    def __init__(
            self,
            logit_scale_init: float = 1 / 0.07,
            learnable: bool = True,
            max_logit_scale: float = 100,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable

        log_logit_scale = torch.ones([]) * np.log(self.logit_scale_init)
        if learnable:
            self.log_logit_scale = nn.Parameter(log_logit_scale)
        else:
            self.register_bufffer("log_logit_scale", log_logit_scale)
    
    def forward(self, x):
        return torch.clip(self.logit_scale.exp(),
                          max=self.max_logit_scale) * x
    
    def extra_repr(self):
        st = f"logit_scale_init={self.logit_scale_init}, learnable={self.learnable}," \
            f"max_logit_scale={self.max_logit_scale}"
        return st
    


class EinOpsRearrange(nn.Module):
    def __init__(self, rearrange_expr: str,
                 **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs
    
    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)
    

def cast_if_src_dtype(
        tensor: torch.Tensor, 
        src_dtype: torch.dtype, 
        tgt_dtype: torch.dtype
):
    updated = False
    if tensor.dtype == src_dtype:
        tensor = tensor.to(dtype=tgt_dtype)
        updated = True
    return tensor, updated


class SelectElements(nn.Module):
    def __init__(self,
                 index) -> None:
        super().__init__()
        self.index = index
    
    def forward(self, x):
        assert x.ndim >= 3
        return x[:, self.index, ...]
    

class SelectEOSAndProject(nn.Module):
    def __init__(self, proj: nn.Module) -> None:
        super().__init__()
        self.proj = proj
    
    def forward(self, x, seq_len):
        assert x.ndim == 3
        x = x[torch.arange(x.shape[0]), seq_len]
        x = self.proj(x)
        return x
    



##################
def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]
    
    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) #dim 21
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def interpolate_pos_encoding_2d(target_spatial_size, pos_embed):
    N = pos_embed.shape[1]
    if N == target_spatial_size:
        return pos_embed
    dim = pos_embed.shape[-1]
    pos_embed, updated = cast_if_src_dtype(pos_embed, torch.bfloat16, torch.float32)
    pos_embed = nn.functional.interpolate(
        pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
        0, 3, 1, 2
        ),
        scale_factor = math.sqrt(target_spatial_size / N),
        mode="bicubic",
    )
    if updated:
        pos_embed, _ = cast_if_src_dtype(pos_embed, torch.float32, torch.bfloat16)
    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return pos_embed

