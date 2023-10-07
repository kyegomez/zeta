import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Layer normalization module.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-5.
        fp16_eps (float, optional): A small value added to the denominator for numerical stability when using fp16 data type. Default is 1e-3.
        stable (bool, optional): Whether to use a stable implementation of layer normalization. Default is False.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        fp16_eps (float): A small value added to the denominator for numerical stability when using fp16 data type.
        stable (bool): Whether to use a stable implementation of layer normalization.
        g (torch.nn.Parameter): The learnable scale parameter.

    # Usage
    ```
    import torch

    # Create an instance of LayerNorm
    layer_norm = LayerNorm(dim=10)

    # Create a random input tensor
    x = torch.randn(32, 10)

    # Apply layer normalization
    normalized_x = layer_norm(x)

    # Print the normalized tensor
    print(normalized_x)

    # Apply L2 normalization
    normalized_x = l2norm(x)

    # Print the normalized tensor
    print(normalized_x)
    ```

    """

    def __init__(self, dim, eps=1e-5, fp16_eps=1e-3, stable=False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass of the layer normalization module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


def l2norm(t):
    """
    L2 normalization function.

    Args:
        t (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The normalized tensor.

    """
    return F.normalize(t, dim=-1)
