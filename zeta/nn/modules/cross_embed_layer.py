from typing import List

import torch
from torch import cat, nn

from zeta.utils.main import default


class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in: int,
        kernel_sizes: List[int],
        dim_out: int = None,
        stride: int = 2,
    ):
        """
        Cross Embed Layer module.

        Args:
            dim_in (int): Input dimension.
            kernel_sizes (List[int]): List of kernel sizes for convolutional layers.
            dim_out (int, optional): Output dimension. Defaults to None.
            stride (int, optional): Stride value for convolutional layers. Defaults to 2.
        """
        super().__init__()
        assert all([(t % 2) == (stride % 2) for t in kernel_sizes])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(
                    dim_in,
                    dim_scale,
                    kernel,
                    stride=stride,
                    padding=(kernel - stride) // 2,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Cross Embed Layer module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return cat(fmaps, dim=1)
