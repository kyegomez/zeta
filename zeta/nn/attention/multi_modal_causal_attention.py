import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class MultiModalCausalAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, visual_features, textual_features, mask=None):
        b, n, _, h = *visual_features.shape, self.heads

        qkv_visual = self.to_qkv(visual_features).chunk(3, dim=-1)
        qkv_textual = self.to_qkv(textual_features).chunk(3, dim=-1)

        q_visual, k_visual, v_visual = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv_visual
        )

        q_textual, k_textual, v_textual = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv_textual
        )

        dots_visual = (
            torch.einsum("bhid,bhjd->bhij", q_visual, k_visual) * self.scale
        )

        dots_textual = (
            torch.einsum(
                "bhid,bhjd->bhij",
                q_textual,
                k_textual,
            )
            * self.scale
        )

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert (
                mask.shape[-1] == dots_textual.shape[-1]
            ), "mask has incorrect dimensions"

            mask = mask[:, None, :] * mask[:, :, None]
            dots_textual.masked_fill(~mask, float("-inf"))

            del mask

        attn_visual = dots_visual.softmax(dim=-1)
        attn_textual = dots_textual.softmax(dim=-1)

        out_visual = torch.einsum(
            "bhij,bhjd->bhid",
            attn_visual,
            v_visual,
        )

        out_textual = torch.einsum(
            "bhij,bhjd->bhid",
            attn_textual,
            v_textual,
        )

        out_visual = rearrange(out_visual, "b h n d -> b n (h d)")

        out_textual = rearrange(out_textual, "b h n d -> b n (h d)")

        return self.to_out(out_visual), self.to_out(out_textual)


class SimpleMMCA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)

        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)

    def forward(self, v, t):
        # self attention for visual tokens
        v = self.self_attn(v, v, v)[0]

        # cross attention for textual tokens
        t = self.cross_attn(t, t, t)[0] + self.cross_attn(t, v, v)[0]

        return t
