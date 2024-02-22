from math import sqrt

import torch
from einops import rearrange
from torch import Tensor, nn


class DepthWiseConv2d(nn.Module):
    def __init__(
        self, dim_in, dim_out, kernel_size, padding, stride, bias=True
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                padding=padding,
                groups=dim_in,
                stride=stride,
                bias=bias,
            ),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


# pooling layer


class Pool(nn.Module):
    def __init__(self, dim: int):
        """
        Pool module that performs pooling operation on input tensors.

        Args:
            dim (int): The input tensor dimension.

        """
        super().__init__()
        self.downsample = DepthWiseConv2d(
            dim, dim * 2, kernel_size=3, stride=2, padding=1
        )
        self.cls_ff = nn.Linear(dim, dim * 2)

    def forward(self, x: Tensor):
        """
        Forward pass of the Pool module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after pooling operation.

        """
        cls_token, tokens = x[:, :1], x[:, 1:]
        cls_token = self.cls_ff(cls_token)
        tokens = rearrange(
            tokens, "b (h w) c -> b c h w", h=int(sqrt(tokens.shape[1]))
        )
        tokens = self.downsample(tokens)
        tokens = rearrange(tokens, "b c h w -> b (h w) c")
        return torch.cat((cls_token, tokens), dim=1)
