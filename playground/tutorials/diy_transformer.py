"""
Zeta was created to build transformer models that can scale limitlessly with an uncompromising
and radically simple user-first API.

We place a strong emphasis on the following:
- modularity
- simplicity
- flexibility
- scalability
- extensibility
- performance

Zeta is built on top of PyTorch and is designed to enable you to build your own models
with extreme reliability.

Let's build an LLM like LLAMA and PALM called Neo
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import pack, unpack
from torch import nn

from zeta.nn import (
    LayerNorm,
    Residual,
    TransformerBlock,
)
from zeta.utils import exists
from zeta.utils.main import eval_decorator, gumnel_sample, top_k


# base model architecture
class Neo(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        causal=True,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        qk_rmsnorm=False,
        lora_r=8,
        rotary_xpos_scale_base=512,
        flash_attn=False,
        finetune_scopes=tuple(),
        cross_entropy_ignore_index=0
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.causal = causal
        self.num_tokens = num_tokens

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            block = Residual(
                TransformerBlock(
                    dim=dim,
                    causal=causal,
                    dim_head=dim_head,
                    heads=heads,
                    qk_rmsnorm=qk_rmsnorm,
                    ff_mult=ff_mult,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    rotary_scale_base=rotary_xpos_scale_base,
                    flash_attn=flash_attn,
                )
            )

            self.layers.append(block)

        self.norm = LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        self.to_logits.weight = self.token_emb.weight

        nn.init.normal_(self.token_emb.weight, std=0.02)

        # loss
        self.cross_entropy_ignore_index = cross_entropy_ignore_index

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path):
        path = Path(path)
        assert path.exists()
        self.load_state_dict(torch.load(str(path)))

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        seq_len,
        prompt=None,
        temperature=1.0,
        filter_logits_fn=top_k,
        filter_thre=0.9,
        pad_value=0.0,
        eos_token=None,
        return_seq_without_prompt=True,
        use_tqdm=False,
        **kwargs
    ):
        if not exists(prompt):
            prompt = torch.zeros(0, self.num_tokens, (1, 1))
            prompt = prompt.to(self.device)
            return_seq_without_prompt = False

        prompt, leading_dims = pack([prompt], "* n")
        n, out = prompt.shape[-1], prompt.clone()

        wrapper_fn = identity if not use_tqdm else quiet_tqdm
        sample_num_times = max(1, seq_len - prompt.shape[-1])

        for _ in wrapper_fn(range(sample_num_times)):
            logits, embed = self.forward(
                out, return_logits_with_embedding=True, **kwargs
            )
            logits, embeds = logits[:, -1], embeds[:, -1]

            if exists(filter_logits_fn):
                logits = filter_logits_fn(logits, thre=filter_thres)

            sample = gumnel_sample(logits, temperature=temperature, dim=-1)

            out, _ = pack([out, sample], "b *")

            if exists(eos_token):
                is_eos_token = out == eos_token

                if is_eos_token.any(dim=-1).all():
                    # MASK OUT EVERYTHING AFTER THE EOS token
                    shifted_is_eos_tokens = F.pad(is_eos_token, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, pad_value)
                    break
        out = unpack(out, leading_dims, "* n ")

        if not return_seq_without_prompt:
            return out

        return out[..., n:]
