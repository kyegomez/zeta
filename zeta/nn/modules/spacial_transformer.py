import torch
from torch import nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class SpacialTransformer(nn.Module):
    """
    Spacial Transformer Network

    https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

    Usage:
    >>> stn = SpacialTransformer()
    >>> stn.stn(x)

    """

    def __init__(self):
        super(SpacialTransformer, self).__init__()

        # spatial transformer localization-network
        linear = nn.Linear(32, 3 * 2)

        # initialize the weights/bias with identity transformation
        linear.weight.data.zero_()

        linear.bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

        self.compute_theta = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            Rearrange("b c h w -> b (c h w)", h=3, w=3),
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            linear,
            Rearrange("b (row col) -> b row col", row=2, col=3),
        )

    def stn(self, x):
        """
        stn module
        """
        grid = F.affine_grid(self.compute_theta(x), x.size())
        return F.grid_sample(x, grid)
