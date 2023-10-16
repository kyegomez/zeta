import torch
from torch import nn
from einops import rearrange


class DynamicPositionBias(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, heads),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        assert j >= i

        rel_dist = torch.arange(j, dtype=torch.float, device=device)
        bias = self.mlp(rearrange(rel_dist, "... -> ... 1"))

        i_seq = torch.arange(j - i, j, device=device)
        j_seq = torch.arange(j, device=device)

        rel_dist_indices = (
            rearrange(i_seq, "i -> i 1") - rearrange(j_seq, "j -> 1 j")
        ).abs()

        bias = rearrange(bias[rel_dist_indices], "i j h -> h i j")
        return bias
