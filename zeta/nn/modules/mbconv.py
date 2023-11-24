import torch
from torch import nn
from einops import reduce, rearrange
from einops import reduce


class DropSample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = (
            torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_()
            > self.prob
        )
        return x + keep_mask / (1 - self.prob)


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            rearrange("b c -> b c 11"),
        )

    def forward(self, x):
        return x + self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.downsample = DropSample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate=4,
    shrinkage_rate=0.25,
    dropout=0.0,
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(
            hidden_dim,
            hidden_dim,
            3,
            stride=stride,
            padding=1,
            groups=hidden_dim,
        ),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out),
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net
