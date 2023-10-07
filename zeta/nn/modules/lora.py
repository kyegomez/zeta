import torch
from torch import nn


class Lora(nn.Module):
    def __init__(self, dim, dim_out, r=8, alpha=None):
        super().__init__()
        self.scale = alpha / r

        self.A = nn.Parameter(torch.randn(dim, r))
        self.B = nn.Parameter(torch.randn(r, dim_out))

    @property
    def weight(self):
        return (self.A @ self.B) * self.scale

    def forward(self, x):
        return x @ self.weight
