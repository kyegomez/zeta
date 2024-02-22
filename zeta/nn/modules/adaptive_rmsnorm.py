import torch.nn.functional as F
from beartype import beartype
from torch import Tensor, nn


def exists(val):
    return val is not None


def append_dims(t, ndims: int):
    return t.reshape(*t.shape, *((1,) * ndims))


class AdaptiveRMSNorm(nn.Module):
    """
    Adaptive Root Mean Square Normalization (RMSNorm) module.

    Args:
        dim (int): The input dimension.
        dim_cond (int): The dimension of the conditioning tensor.
        channel_first (bool, optional): Whether the input has channels as the first dimension. Defaults to False.
        images (bool, optional): Whether the input represents images. Defaults to False.
        bias (bool, optional): Whether to include a bias term. Defaults to False.
    """

    def __init__(
        self, dim, *, dim_cond, channel_first=False, images=False, bias=False
    ):
        super().__init__()

        self.dim_cond = dim_cond
        self.channel_first = channel_first
        self.scale = dim**0.5

        self.to_gamma = nn.Linear(dim_cond, dim)
        self.to_bias = nn.Linear(dim_cond, dim) if bias else None

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        if bias:
            nn.init.zeros_(self.to_bias.weight)
            nn.init.zeros_(self.to_bias.bias)

    @beartype
    def forward(self, x: Tensor, *, cond: Tensor):
        """
        Forward pass of the AdaptiveRMSNorm module.

        Args:
            x (torch.Tensor): The input tensor.
            cond (torch.Tensor): The conditioning tensor.

        Returns:
            torch.Tensor: The normalized and conditioned output tensor.
        """
        batch = x.shape[0]
        assert cond.shape == (batch, self.dim_cond)

        gamma = self.to_gamma(cond)

        bias = 0.0
        if exists(self.to_bias):
            bias = self.to_bias(cond)

        if self.channel_first:
            gamma = append_dims(gamma, x.ndim - 2)

            if exists(self.to_bias):
                bias = append_dims(bias, x.ndim - 2)

        return (
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale
            * gamma
            + bias
        )
