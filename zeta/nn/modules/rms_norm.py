import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        groups=1
    ):
        super().__init__()
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.ones(groups, dim, 1))

    def forward(self, x):
        normed = F.normalize(x, dim=-2)
        return normed * self.scale * self.gamma
