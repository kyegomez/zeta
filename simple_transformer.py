import torch
from torch import nn
from zeta.nn.modules.feedforward import FeedForward
from zeta.nn.attention.shaped_attention import ShapedAttention
from zeta.nn.modules.residual import Residual
from zeta.nn.attention import FlashAttention


class SimpleTransformerBlock(nn.Module):
    """
    Simple Transformer Block

    Args:
        dim (int): Input dimension
        depth (int): Depth of the transformer
        heads (int): Number of heads
        dropout (float): Dropout probability

    Usage:
    >>> model = SimpleTransformerBlock(768, 12, 8, 0.1)
    >>> x = torch.randn(1, 768)
    >>> model(x).shape

    """

    def __init__(
        self,
        dim,
        depth,
        heads,
        dropout: float = 0.0,
    ):
        super(SimpleTransformerBlock, self).__init__()
        self.layers = nn.ModuleList([])
        self.x_proj = nn.Linear(dim, dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ShapedAttention(dim, heads, dropout=dropout),
                        FeedForward(
                            dim,
                            dim,
                            dropout=dropout,
                            # relu_squared=True,
                            # post_act_ln=True,
                        ),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        """
        x -> x_proj -> attn -> matmul with x -> ff -> out + x

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor



        """
        x_for_matmul = self.x_proj(x)

        for attn, ff in self.layers:
            attn = attn(x)
            matmul = torch.matmul(attn, x_for_matmul)
            out = ff(x) + matmul
        return out


# transformer
def SimpleTransformer(
    *,
    dim,
    num_tokens,
    depth,
    dim_head=64,
    heads=8,
):
    """
    Simple Transformer

    Args:
        dim (int): Input dimension
        num_tokens (int): Number of tokens
        depth (int): Depth of the transformer
        dim_head (int): Dimension of the head
        heads (int): Number of heads

    Usage:
    >>> model = SimpleTransformer(768, 20000, 12, 64, 8)
    >>> x = torch.randint(0, 20000, (1, 768))
    >>> model(x).shape



    """
    net = nn.Sequential(
        nn.Embedding(num_tokens, dim),
        *[
            Residual(
                SimpleTransformerBlock(dim, depth, heads, dropout=0.1),
            )
            for _ in range(depth)
        ],
        nn.Linear(dim, num_tokens, bias=False),
    )

    nn.init.normal_(net[0].weight, std=0.02)
    return net


tokens = torch.randint(0, 20000, (1, 2048))
model = SimpleTransformer(dim=2048, num_tokens=20000, depth=12, heads=8)
out = model(tokens)
print(out)
