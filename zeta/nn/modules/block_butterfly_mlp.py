from typing import List

import torch
from torch import Tensor, nn


class BlockButterflyLinear(nn.Module):
    """
    BlockButterflyMLP is a module that applies a block butterfly transformation to the input tensor.

    Args:
        num_blocks (int): The number of blocks in the butterfly transformation.
        input_block_dim (int): The dimension of each input block.
        output_block_dim (int): The dimension of each output block.
    """

    def __init__(
        self,
        num_blocks: int,
        input_block_dim: int,
        output_block_dim: int,
    ):
        super().__init__()
        self.weight = torch.randn(num_blocks, input_block_dim, output_block_dim)
        self.bias = torch.randn(num_blocks, 1, output_block_dim)

    def forward(self, x: Tensor):
        return torch.batch_matmul(x, self.weight) + self.bias


class BlockMLP:
    def __init__(
        self,
        dim: int,
        layer_block_dims: List[int],
        layer_dims: List[int],
        act=nn.GELU(),
    ):
        """
        Initializes a BlockMLP module.

        Args:
            dim (int): The input dimension.
            layer_block_dims (List[int]): The dimensions of each block in the MLP.
            layer_dims (List[int]): The dimensions of each layer in the MLP.
            act (nn.Module, optional): The activation function to be applied after each block. Defaults to nn.GELU().
        """
        super().__init__()
        self.dim = dim
        self.layer_block_dims = layer_block_dims
        self.act = act

        self.block_dim = layer_dims
        num_blocks = dim // layer_block_dims[0]

        # Create block mlp
        self.mlp = nn.Sequential([])
        for i in range(len(layer_block_dims) - 1):
            self.mlp += [
                BlockButterflyLinear(
                    num_blocks, layer_block_dims[i], layer_block_dims[i + 1]
                ),
                act,
            ]

        self.mlp = self.mlp[:-1]

    def forward(self, x: Tensor):
        """
        Forward pass of the BlockMLP module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        bs, input_dim = x.shape
        x = x.view(bs, -1, self.block_dim).tranpose(0, 1)
        x = self.mlp(x)
        x = x.tranpose(1, 0).view(bs, -1)
        return x
