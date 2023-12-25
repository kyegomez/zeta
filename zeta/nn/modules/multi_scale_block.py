import torch
from torch import nn
import torch.nn.functional as F


class MultiScaleBlock(nn.Module):
    """
    A module that applies a given submodule to the input tensor at multiple scales.

    Args:
        module (nn.Module): The submodule to apply.

    Returns:
        torch.Tensor: The output tensor after applying the submodule at multiple scales.
    """

    def __init__(self, module):
        super().__init__()
        self.submodule = module

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x1 = F.interpolate(x, scale_factor=0.5, *args, **kwargs)
        x2 = F.interpolate(x, scale_factor=2.0, *args, **kwargs)
        return (
            self.submodule(x)
            + F.interpolate(self.submodule(x1), size=x.shape[2:])
            + F.interpolate(self.submodule(x2), size=x.shape[2:])
        )
