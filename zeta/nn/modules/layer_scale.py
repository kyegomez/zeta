from torch.nn import Module
import torch 
from torch import nn, Tensor

class LayerScale(Module):
    """
    Applies layer scaling to the output of a given module.

    Args:
        fn (Module): The module to apply layer scaling to.
        dim (int): The dimension along which to apply the scaling.
        init_value (float, optional): The initial value for the scaling factor. Defaults to 0.

    Attributes:
        fn (Module): The module to apply layer scaling to.
        gamma (Parameter): The scaling factor parameter.

    """

    def __init__(self, fn: Module, dim, init_value=0.):
        super().__init__()
        self.fn = fn
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)

        if isinstance(out, Tensor):
            return out * self.gamma

        out, *rest = out
        return out * self.gamma, *rest