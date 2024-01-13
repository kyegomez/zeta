from torch import nn
from torch.nn import Module

from zeta.nn.modules.rms_norm import RMSNorm


class PreNorm(Module):
    """
    Pre-normalization module that applies RMSNorm to the input before passing it through the given function.

    Args:
        dim (int): The dimension of the input.
        fn (Module): The function to apply to the normalized input.

    Attributes:
        fn (Module): The function to apply to the normalized input.
        norm (RMSNorm): The RMSNorm instance used for normalization.
    """

    def __init__(self, dim, fn: Module):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, **kwargs):
        """
        Forward pass of the PreNorm module.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            torch.Tensor: The output tensor after applying the function to the normalized input and adding the input tensor.
        """
        return self.fn(self.norm(x), **kwargs) + x

class PostNorm(Module):
    """
    Post-normalization module that applies layer normalization after the input is passed through a given module.

    Args:
        dim (int): The dimension of the input tensor.
        fn (Module): The module to be applied to the input tensor.

    Attributes:
        fn (Module): The module to be applied to the input tensor.
        norm (LayerNorm): The layer normalization module.

    """

    def __init__(self, dim, fn: Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        """
        Forward pass of the PostNorm module.

        Args:
            x (Tensor): The input tensor.
            **kwargs: Additional keyword arguments to be passed to the underlying module.

        Returns:
            Tensor: The output tensor after applying the post-normalization.

        """
        return self.norm(self.fn(x, **kwargs) + x)
