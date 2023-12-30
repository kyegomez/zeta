import math

from einops import rearrange
from torch import einsum, nn

from zeta.utils import l2norm


class LinearAttention(nn.Module):
    """
    Linear Attention module that performs attention mechanism on the input feature map.

    Args:
        dim (int): The input feature map dimension.
        dim_head (int, optional): The dimension of each attention head. Defaults to 32.
        heads (int, optional): The number of attention heads. Defaults to 8.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The output feature map after applying linear attention.

    """

    def __init__(self, dim: int, dim_head: int = 32, heads: int = 8, **kwargs):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)

        self.nonlin = nn.GELU()
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias=False), nn.LayerNorm(dim)
        )

    def forward(self, fmap):
        """
        Forward pass of the LinearAttention module.

        Args:
            fmap (torch.Tensor): Input feature map tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying linear attention, of shape (batch_size, channels, height, width).
        """
        h, x, y = self.heads, *fmap.shape[-2:]
        seq_len = x * y

        fmap = self.norm(fmap)
        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> (b h) (x y) c", h=h),
            (q, k, v),
        )

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale
        v = l2norm(v)

        k, v = map(lambda t: t / math.sqrt(seq_len), (k, v))

        context = einsum("b n d, b n e -> b d e", k, v)
        out = einsum("b n d, b d e -> b n e", q, context)
        out = rearrange(out, "(b h) (x y) d -> b (h d) x y", h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)

