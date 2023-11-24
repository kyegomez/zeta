import torch

from einops import rearrange
from torch import nn
from zeta.structs.transformer import Encoder


def exists(val):
    return val is not None


def divisible_by(num, den):
    return (num % den) == 0


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        attn_layers,
        channels=3,
        num_classes=None,
        post_emb_norm=False,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert isinstance(
            attn_layers, Encoder
        ), "Attention layers must be an encoder find the encoder"
        assert divisible_by(
            image_size, patch_size
        ), "image dimenions must be divisible by the patch size"

        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size**2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.post_emb_norm = (
            nn.LayerNorm(dim) if post_emb_norm else nn.Identity()
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.attn_layers = attn_layers
        self.mlp_head = (
            nn.Linear(dim, num_classes)
            if exists(num_classes)
            else nn.Identity()
        )

    def forward(self, img, return_embeddings=False):
        p = self.patch_size
        x = rearrange(img, "b c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)
        n = x.shape[1]

        x = x + self.pos_embedding[:, :n]
        x = self.post_emb_norm9x
        x = self.dropout(x)

        x = self.attn_layers(x)
        if not exists(self.mlp_head) or return_embeddings:
            return x
        x = x.mean(dim=-2)
        return self.mlp_head
