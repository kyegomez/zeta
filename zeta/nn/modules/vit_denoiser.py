import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn


def to_patch_embedding(x: Tensor, patch_size: int, patch_dim: int, dim):
    """
    Converts the input tensor into patch embeddings.

    Args:
        x (Tensor): The input tensor.
        patch_size (int): The size of each patch.
        patch_dim (int): The dimension of each patch.
        dim: The output dimension of the patch embedding.

    Returns:
        Tensor: The patch embedding tensor.
    """
    return nn.Sequential(
        Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        ),
        nn.LayerNorm(patch_dim),
        nn.Linear(patch_dim, dim),
        nn.LayerNorm(dim),
    )


def posemb_sincos_2d(
    patches,
    temperature: int = 10000,
    dtype=torch.float32,
):
    """
    Computes positional embeddings using sine and cosine functions for a 2D grid.

    Args:
        patches (torch.Tensor): Input patches of shape (batch_size, height, width, dim).
        temperature (int, optional): Temperature parameter for the positional embeddings. Defaults to 10000.
        dtype (torch.dtype, optional): Data type of the positional embeddings. Defaults to torch.float32.

    Returns:
        torch.Tensor: Positional embeddings of shape (batch_size, height * width, dim).

    Raises:
        AssertionError: If the feature dimension is not a multiple of 4.
    """
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.mesgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    assert (
        dim % 4
    ) == 0, "feature dimension must be a multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class VisionAttention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.1
    ):
        """
        VisionAttention module performs self-attention on the input tensor.

        Args:
            dim (int): The input dimension of the tensor.
            heads (int, optional): The number of attention heads. Defaults to 8.
            dim_head (int, optional): The dimension of each attention head. Defaults to 64.
            dropout (float, optional): The dropout probability. Defaults to 0.1.

        Example::
            >>> x = torch.randn(1, 3, 32, 32)
            >>> model = VisionAttention(dim=32, heads=8, dim_head=64, dropout=0.1)
            >>> out = model(x)
            >>> print(out)
        """
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the VisionAttention module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after self-attention.
        """
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b p n (h d) -> b h p n d", h=self.heads),
            qkv,
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b p h n d -> b p n (h d)")
        return self.to_out(out)


class VitTransformerBlock(nn.Module):
    """
    Transformer block used in the Vision Transformer (ViT) denoiser model.

    Args:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mlp_dim (int): The dimension of the feed-forward network.
        expansion (int): The expansion factor for the feed-forward network.
        dropout (float): The dropout rate.

    Attributes:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mlp_dim (int): The dimension of the feed-forward network.
        expansion (int): The expansion factor for the feed-forward network.
        dropout (float): The dropout rate.
        norm (nn.LayerNorm): Layer normalization module.
        attn (VisionAttention): VisionAttention module for self-attention.
        mlp (nn.Sequential): Feed-forward network module.

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        expansion: int,
        dropout: float,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.expansion = expansion
        self.dropout = dropout

        self.norm = nn.LayerNorm(dim)
        self.attn = VisionAttention(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim * expansion),
            nn.GELU(),
            nn.Linear(mlp_dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the VitTransformerBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        x = self.norm(x)
        x = self.attn(x) + x
        x = self.mlp(x) + x

        return x
