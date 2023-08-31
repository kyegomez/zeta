#from paper:: https://arxiv.org/pdf/2308.10882.pdf

import torch
from torch import nn
from einops import rearrange

def exists(val):
    return val is not None

class TruncatedRotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim,
            a,
            b,
            rho
    ):
        super().__init__()
        self.dim = dim
        self.a = a 
        self.b = b
        self.rho = rho
        self.base = 10000
        self.inv_freq = 1. / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', self.inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i, j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        theta = self.base ** (-2 * torch.arange(0, self.dim, 2).float() / self.dim)
        theta_star = torch.where(theta >= self.b, theta, 
                                 torch.where(theta < self.a, torch.zeros_like(theta), self.rho * torch.ones_like(theta)))
        theta_star = torch.cat((theta_star, theta_star), dim=-1)

        result = freqs * theta_star
        return result
