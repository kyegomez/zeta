from torch import nn, Tensor, einsum
from einops import rearrange
from zeta.utils.main import exists


class LinearAttention(nn.Module):
    """
    LinearAttention module performs linear attention mechanism on the input tensor.

    Args:
        dim (int): The dimension of the input tensor.
        heads (int, optional): The number of attention heads. Defaults to 4.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout probability. Defaults to 0.0.

    Returns:
        Tensor: The output tensor after linear attention mechanism.


    Example::
        >>> import torch
        >>> from zeta.nn.attention import LinearAttention
        >>> x = torch.randn(1, 32, 64)
        >>> attn = LinearAttention(64)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([1, 32, 64])
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 64,
        dropout: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout

        inner_dim = heads * dim_head
        self.scale = dim_head**-0.5

        # Linear projection layers
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: Tensor, mask: Tensor = None):
        """
        Perform forward pass of the LinearAttention module.

        Args:
            x (Tensor): The input tensor.
            mask (Tensor, optional): The mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor after linear attention mechanism.
        """
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
        )

        q = q * self.scale
        q, k = q.softmax(dim=-1), k.softmax(dim=-2)

        if exists(mask):
            k.masked_fill(mask, 0.0)

        context = einsum("b n d, b n e -> b d e", q, k)
        out = einsum("b d e, b n d -> b n e", context, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)
