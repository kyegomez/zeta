import torch
from torch import nn, Tensor
from zeta.nn.modules.chan_layer_norm import ChanLayerNorm
from einops import rearrange


class ScalableImgSelfAttention(nn.Module):
    """
    ScalableImgSelfAttention module applies self-attention mechanism to image data.

    Args:
        dim (int): The input dimension of the image.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_key (int, optional): The dimension of the key vectors. Defaults to 32.
        dim_value (int, optional): The dimension of the value vectors. Defaults to 32.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        reduction_factor (int, optional): The reduction factor for downscaling the image. Defaults to 1.

    Attributes:
        dim (int): The input dimension of the image.
        heads (int): The number of attention heads.
        dim_key (int): The dimension of the key vectors.
        dim_value (int): The dimension of the value vectors.
        reduction_factor (int): The reduction factor for downscaling the image.
        scale (float): The scaling factor for the key vectors.
        attend (nn.Softmax): The softmax function for attention calculation.
        dropout (nn.Dropout): The dropout layer.
        norm (ChanLayerNorm): The channel-wise layer normalization.
        to_q (nn.Conv2d): The convolutional layer for query projection.
        to_k (nn.Conv2d): The convolutional layer for key projection.
        to_v (nn.Conv2d): The convolutional layer for value projection.
        to_out (nn.Sequential): The sequential layer for output projection.

    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_key: int = 32,
        dim_value: int = 32,
        dropout: float = 0.0,
        reduction_factor: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.reduction_factor = reduction_factor

        self.scale = dim_key**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.norm = ChanLayerNorm(dim)

        # Projections
        self.to_q = nn.Conv2d(dim, dim_key * heads, 1, bias=False)
        self.to_k = nn.Conv2d(
            dim,
            dim_key * heads,
            reduction_factor,
            stride=reduction_factor,
            bias=False,
        )
        self.to_v = nn.Conv2d(
            dim,
            dim_value * heads,
            reduction_factor,
            stride=reduction_factor,
            bias=False,
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(dim_value * heads, dim, 1), nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ScalableImgSelfAttention module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: The output tensor of shape (batch_size, channels, height, width).

        """
        h, w, h = *x.shape[-2:], self.heads

        x = self.norm(x)

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # Split out heads
        q, k, v = map(
            lambda t: rearrange(t, "b (h d) ... -> b h (...) d", h=h),
            (
                q,
                k,
            ),
        )

        # Similarity
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Attention
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Aggregate values
        out = torch.matmul(attn, v)

        # Merge back heads
        out = rearrange(
            out,
            "b h (x y) d -> b (h d) x y",
            x=h,
            y=w,
        )
        return self.to_out(out)


# x = torch.randn(1, 3, 64, 64)
# peg = ScalableImgSelfAttention(3)
# out = peg(x)
# print(out.shape)
