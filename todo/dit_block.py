import torch
from torch import nn, Tensor
from zeta.nn.attention.multiquery_attention import MultiQueryAttention
from zeta.nn.modules.feedforward import FeedForward
from zeta.nn.modules.scale import Scale
from zeta.nn.modules.adaptive_layernorm import AdaptiveLayerNorm


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) module.

    Args:
        dim (int): The input dimension.
        eps (float): A small value added to the denominator for numerical stability.
        scale (int): The scale factor for the linear layer.
        bias (bool): Whether to include a bias term in the linear layer.
    """

    def __init__(
        self,
        dim: int = None,
        eps: float = 1e-5,
        scale: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.scale = scale
        self.bias = bias

        self.norm = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * scale, bias=bias),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the AdaLN module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The normalized output tensor.
        """
        return self.norm(x)


class DitBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = None,
        dropout: float = 0.1,
        heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.dropout = dropout
        self.heads = heads

        # Attention
        self.attn = MultiQueryAttention(
            dim,
            heads,
        )

        # FFN
        self.input_ffn = FeedForward(dim, dim, 4, swish=True)

        # Conditioning mlp
        self.conditioning_mlp = FeedForward(dim, dim, 4, swish=True)

        # Shift
        # self.shift_op = ShiftTokens()

        # Norm
        self.norm = AdaptiveLayerNorm(dim)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:

        # Norm
        self.norm(x)

        # Scale
        # scaled = modulate(
        #     x,
        #     normalize,
        #     normalize
        # )

        # return scaled
        scaled = Scale(fn=self.norm)(x)
        return scaled


input = torch.randn(1, 10, 512)
conditioning = torch.randn(1, 10, 512)
dit_block = DitBlock(512)
output = dit_block(input, conditioning)
print(output.shape)
