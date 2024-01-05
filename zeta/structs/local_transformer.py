import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from zeta.nn.attention.local_attention_mha import LocalMHA
from zeta.nn.biases.dynamic_position_bias import DynamicPositionBias
from zeta.nn.modules import feedforward_network
from zeta.utils.main import eval_decorator, exists, top_k


class LocalTransformer(nn.Module):
    """
    LocalTransformer module that implements a local self-attention transformer.

    Args:
        num_tokens (int): The number of tokens in the input vocabulary.
        max_seq_len (int): The maximum sequence length.
        dim (int): The dimensionality of the token and positional embeddings.
        depth (int): The number of transformer layers.
        causal (bool, optional): Whether to use causal attention. Defaults to True.
        local_attn_window_size (int, optional): The size of the local attention window. Defaults to 512.
        dim_head (int, optional): The dimensionality of each attention head. Defaults to 64.
        heads (int, optional): The number of attention heads. Defaults to 8.
        ff_mult (int, optional): The multiplier for the feedforward network dimension. Defaults to 4.
        attn_dropout (float, optional): The dropout rate for attention layers. Defaults to 0.0.
        ff_dropout (float, optional): The dropout rate for feedforward layers. Defaults to 0.0.
        ignore_index (int, optional): The index to ignore during loss calculation. Defaults to -1.
        use_xpos (bool, optional): Whether to use positional embeddings based on xpos. Defaults to False.
        xpos_scale_base (None, optional): The base value for scaling xpos positional embeddings. Defaults to None.
        use_dynamic_pos_bias (bool, optional): Whether to use dynamic positional bias. Defaults to False.

    Attributes:
        token_emb (nn.Embedding): Embedding layer for token embeddings.
        pos_emb (nn.Embedding): Embedding layer for positional embeddings.
        max_seq_len (int): The maximum sequence length.
        layers (nn.ModuleList): List of transformer layers.
        local_attn_window_size (int): The size of the local attention window.
        dynamic_pos_bias (DynamicPositionBias or None): Dynamic positional bias layer, if enabled.
        ignore_index (int): The index to ignore during loss calculation.
        to_logits (nn.Sequential): Sequential layer for converting transformer output to logits.
    """

    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        causal=True,
        local_attn_window_size=512,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ignore_index=-1,
        use_xpos=False,
        xpos_scale_base=None,
        use_dynamic_pos_bias=False,
        **kwargs,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len
        self.layers = nn.ModuleList([])

        self.local_attn_window_size = local_attn_window_size
        self.dynamic_pos_bias = None
        if use_dynamic_pos_bias:
            self.dynamic_pos_bias = DynamicPositionBias(
                dim=dim // 2, heads=heads
            )

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        LocalMHA(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            causal=causal,
                            window_size=local_attn_window_size,
                            use_xpos=use_xpos,
                            xpos_scale_base=xpos_scale_base,
                            use_rotary_pos_emb=not use_dynamic_pos_bias,
                            prenorm=True,
                            **kwargs,
                        ),
                        feedforward_network(
                            dim=dim, mult=ff_mult, dropout=ff_dropout
                        ),
                    ]
                )
            )

        self.ignore_index = ignore_index
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, num_tokens, bias=False)
        )

    @torch.no_grad()
    @eval_decorator
    def generate(
        self, prime, seq_len, temperature=1.0, filter_thres=0.9, **kwargs
    ):
        # einops conflicts with ruff, so noqa on next line
        n, device = prime.shape[1], prime.device  # noqa F841

        out = prime

        for _ in range(seq_len):
            logits = self.forward(out[:, -self.max_seq_len :], **kwargs)
            filtered_logits = top_k(logits[:, -1], thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sampled = torch.multinomial(probs, 1)
            out = torch.cat((out, sampled), dim=-1)

        return out[:, n:]

    def forward(self, x, mask=None, return_loss=False):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        assert n <= self.max_seq_len
        x = x + self.pos_emb(torch.arange(n, device=device))

        # dynamic pos bias

        attn_bias = None
        if exists(self.dynamic_pos_bias):
            w = self.local_attn_window_size
            attn_bias = self.dynamic_pos_bias(w, w * 2)

        # go through layers

        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_bias=attn_bias) + x
            x = ff(x) + x

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        logits = rearrange(logits, "b n c -> b c n")
        loss = F.cross_entropy(logits, labels, ignore_index=self.ignore_index)
        return loss
