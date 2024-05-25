"""
Nirvana

Multi grouped query attention + feedforward


"""

import torch
from torch import Tensor, nn

from zeta.nn import FeedForward, OutputHead
from zeta.nn.attention import MultiQueryAttention


class TransformerBlock(nn.Module):
    """
    TransformerBlock is a module that represents a single block in a transformer model.

    Args:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads.
        mult (int): The multiplier for the hidden dimension in the feed-forward network.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, dim: int, heads: int, mult: int, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.mult = mult

        # Multi-grouped query attention
        self.attn = MultiQueryAttention(dim, heads, *args, **kwargs)

        # Ffn
        self.ffn = FeedForward(dim, dim, mult, swish=True, post_act_ln=True)

        # LayerNorm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the TransformerBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the TransformerBlock.
        """
        skip = x

        x = self.norm(x)

        # Attn
        x, _, _ = self.attn(x)
        x + skip

        # ffn
        skip_two = x

        # Ffn
        return self.ffn(x) + skip_two


class Nirvna(nn.Module):
    """
    A class representing the Nirvna model.

    Args:
        dim (int): The dimension of the model.
        heads (int): The number of attention heads.
        mult (int): The multiplier for the hidden dimension in the feed-forward network.
        depth (int, optional): The number of transformer blocks. Defaults to 8.
        num_tokens (int, optional): The number of tokens in the input vocabulary. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The dimension of the model.
        heads (int): The number of attention heads.
        mult (int): The multiplier for the hidden dimension in the feed-forward network.
        depth (int): The number of transformer blocks.
        num_tokens (int): The number of tokens in the input vocabulary.
        embed (nn.Embedding): The embedding layer.
        layers (nn.ModuleList): The list of transformer blocks.

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mult: int,
        depth: int = 8,
        num_tokens: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.mult = mult
        self.depth = depth
        self.num_tokens = num_tokens

        # Embedding
        self.embed = nn.Embedding(num_tokens, dim)

        # Layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, heads, mult, *args, **kwargs)
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        """
        Forward pass of the Nirvna model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.

        """
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x)

        x = OutputHead(self.dim, -1)(x)
        return x


# Forward pass
x = torch.randint(0, 100, (1, 100))


# Model
model = Nirvna(512, 8, 4, 8, 100)

# Forward
y = model(x)
print(y)
