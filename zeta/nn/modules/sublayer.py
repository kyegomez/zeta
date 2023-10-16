from torch import nn


class LN(nn.Module):
    def __init__(self, dim=None, eps=None):
        self.dim = dim
        self.eps = eps

    def forward(self):
        nn.LayerNorm(self.dim, self.eps)


def subln(x):
    return x + LN(x)
