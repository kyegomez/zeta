import torch
from einops import rearrange
from torch import nn

from zeta.structs.attn_layers import Attention, RotaryEmbedding
from zeta.structs.parallel_transformer import SwiGLU
from zeta.nn.embeddings.xpos_relative_position import apply_rotary_pos_emb
from zeta.nn.modules.layernorm import LayerNorm
from zeta.utils.main import exists, l2norm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        causal=True,
        heads=8,
        qk_rmsnorm=False,
        qk_scale=8,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        use_xpos=True,
        xpos_scale_base=512,
        flash_attn=False,
    ):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (
            attn_inner_dim,
            dim_head,
            dim_head,
            (ff_inner_dim * 2),
        )

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.attend = Attention(
            causal=causal, dropout=attn_dropout, use_flash_attn=flash_attn
        )

        self.heads = heads
        self.scale = (dim_head**-0.5) if not qk_rmsnorm else qk_scale
        self.causal = causal

        self.rotary_emb = RotaryEmbedding(
            dim_head, scale_base=xpos_scale_base, use_xpos=use_xpos and causal
        )

        self.fused_attn_ff_proj = nn.Linear(
            dim, sum(self.fused_dims), bias=False
        )

        self.flash_attn = flash_attn
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.flash_attn_dropout = attn_dropout

        # parallel feedforward tail

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_inner_dim, dim, bias=False),
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("pos_emb", None, persistent=False)
        self.register_buffer("pos_emb_scale", None, persistent=False)

    def get_rotary_embedding(self, n, device):
        if exists(self.pos_emb) and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n], self.pos_emb_scale[:n]

        pos_emb, scale = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        self.register_buffer("pos_emb_scale", scale, persistent=False)
        return pos_emb, scale

    def forward(self, x, mask=None, finetune_modules=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # finetune loras

        lora_q = lora_k = lora_v = lora_o = None

        if exists(finetune_modules):
            lora_q, lora_k, lora_v, lora_o = finetune_modules
            q = q + lora_q(x)
            k = k + lora_k(x)
            v = v + lora_v(x)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # qk rmsnorm

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        # rotary embeddings with xpos decay for better length extrapolation

        positions, scale = self.get_rotary_embedding(n, device)

        q = apply_rotary_pos_emb(positions, q, scale)
        k = apply_rotary_pos_emb(positions, k, scale**-1)

        # attention function, either regular or flash

        out = self.attend(q, k, v, mask=mask)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")

        attn_out = self.attn_out(out)

        ff_out = self.ff_out(ff)

        if exists(lora_o):
            attn_out = attn_out + lora_o(out)

        return attn_out + ff_out


# transformer
