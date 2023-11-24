import torch
from torch import nn

from zeta.utils.main import exists, l2norm


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed=False):
        super().__init__()
        self.scale = dim**-0.5 if not l2norm_embed else 1.0
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None):
        seq_len, device = x.shape[-1], x.device
        assert seq_len <= self.max_seq_len, (
            f"You are passing in a sequence length of {seq_len} but you"
            " absolute positional embedding has a max of length of"
            f" {self.max_seq_len}"
        )

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb
