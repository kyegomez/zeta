import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class LayerNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps, elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim_head)
        self.norm_v = nn.LayerNorm(dim_head)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        # #normalize key and values,  QK Normalization
        k = self.norm_k(k)
        v = self.norm_v(v)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim, heads=heads, dim_head=dim_head, dropout=dropout
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            # layernorm before attention
            x = self.norm(x)

            # parallel
            x = x + attn(x) + ff(x)

        return self.norm(x)


class MegaVit(nn.Module):
    """
    MegaVit model from https://arxiv.org/abs/2106.14759

    Args:
    -------
    image_size: int
        Size of image
    patch_size: int
        Size of patch
    num_classes: int
        Number of classes
    dim: int
        Dimension of embedding
    depth: int
        Depth of transformer
    heads: int
        Number of heads
    mlp_dim: int
        Dimension of MLP
    pool: str
        Type of pooling
    channels: int
        Number of channels
    dim_head: int
        Dimension of head
    dropout: float
        Dropout rate
    emb_dropout: float
        Dropout rate for embedding

    Returns:
    --------
    torch.Tensor
        Predictions

    Methods:
    --------
    forward(img: torch.Tensor) -> torch.Tensor:
        Forward pass

    Architecture:
    -------------
    1. Input image is passed through a patch embedding layer
    2. Positional embedding is added
    3. Dropout is applied
    4. Transformer is applied
    5. Pooling is applied
    6. MLP head is applied
    7. Output is returned

    Usage
    -----
    >>> model = MegaVit(
    ...     image_size = 256,
    ...     patch_size = 32,
    ...     num_classes = 1000,
    ...     dim = 512,
    ...     depth = 6,
    ...     heads = 8,
    ...     mlp_dim = 1024,
    ...     dropout = 0.1,
    ...     emb_dropout = 0.1
    ... )
    >>> img = torch.randn(1, 3, 256, 256)
    >>> preds = model(img) # (1, 1000)

    References:
    -----------
    [1] https://arxiv.org/abs/2106.14759

    """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (
            image_width // patch_width
        )
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
