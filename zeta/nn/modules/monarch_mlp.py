import torch
from torch import nn, Tensor


class MonarchMLP(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.glu = nn.GLU()
        self.gelu = nn.GELU()

    def forward(self, x: Tensor):
        x = self.glu(x)
        x = self.gelu(x)
        return x
