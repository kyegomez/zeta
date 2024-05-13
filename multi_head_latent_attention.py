import torch
from torch import nn, Tensor
from zeta.nn.embeddings.rope import RotaryEmbedding
from zeta.nn.attention.multiquery_attention import MultiQueryAttention


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        hidden_dim: int = None,
        rope: bool = False,
        rope_scale_base: int = 512,
        batch_size: int = 1,
        seqlen: int = 10000,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.rope = rope
        self.rope_scale_base = rope_scale_base
        self.batch_size = batch_size
        self.seqlen = seqlen

        # Rotary Embedding
        self.rope = RotaryEmbedding(
            dim, use_xpos=True, scale_base=rope_scale_base, *args, **kwargs
        )

        # Attention
        self.attn = MultiQueryAttention(dim, heads, *args, **kwargs)

        #
        self.latent_q = nn.Parameter(torch.randn(batch_size, seqlen, dim))

        # KV
        self.latent_kv = nn.Parameter(torch.randn(batch_size, seqlen, dim))

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        k_r_t, scale = self.rope(self.seqlen, device)
        print(k_r_t)
        x = k_r_t + x


# # Example
# x = torch.randn(1, 100, 10)

# # Attention
# model = MultiHeadLatentAttention(
#     10,
#     8,
# )

# # Apply the model
# out = model(x)
# print(out.shape)
