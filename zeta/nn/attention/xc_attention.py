from torch import nn, einsum
from einops import rearrange, pack_one, unpack_one
import torch.nn.functional as F
from einops.layers.torch import Rearrange


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, dim=-1)


class XCAttention(nn.Module):
    """
    From XCiT: Cross-Covariance Image Transformers

    Args:
        dim (int): Number of input channels
        cond_dim (int): Number of conditioning channels
        dim_head (int): Number of channels per head
        heads (int): Number of attention heads
        scale (int): Scale of attention
        flash (bool): Whether to use FLASH attention
        dropout (float): Dropout rate

    Returns:
        Tensor: Output tensor

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

            >>> import torch
            >>> from zeta.nn.attention import XCAttention
            >>> self_attn = XCAttention(dim=256, heads=8)
            >>> x = torch.randn(1, 256, 16, 16)
            >>> out = self_attn(x) # 1x256x16x16


    """

    def __init__(
        self,
        *,
        dim,
        cond_dim: int,
        dim_head: int = 32,
        heads: int = 8,
        scale: int = 8,
        flash=False,
        dropout: 0.0,
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.has_cond = exists(cond_dim)
        self.film = None

        if self.has_cond:
            self.film = nn.Sequential(
                nn.Linear(cond_dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim_inner),
                Rearrange("b (r d) -> r b 1 d", r=2),
            )

        self.nrom = nn.LayerNorm(dim, elementwise_affine=not self.has_cond)
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b h d n -> b n (h d)"),
            nn.Linear(dim_inner, dim),
        )

    def forward(self, x, cond=None):
        """
        Forward pass

        Args:
            x (Tensor): Input tensor
            cond (Tensor): Conditioning tensor

        Returns:
            Tensor: Output tensor

        Shape:
            - Input: :math:`(B, C, H, W)`
            - Output: :math:`(B, C, H, W)`

        """
        x = rearrange(x, "b c h w -> b h w c")
        x, ps = pack_one(x, "b * c ")
        x = self.norm(x)

        # conditioning
        if exists(self.film):
            assert exists(cond)

            gamma, beta = self.film(cond)
            x = x * gamma + beta

        # Cosine sim linear attention
        q, k, v = self.to_qkv(x)
        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        sim = einsum("b h i n, b h j n -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j n -> b h i n", attn, v)
        out = self.to_out(out)
        out = unpack_one(out, ps, "b * c")
        return rearrange(out, "b h w c -> b c h w")
