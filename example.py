import torch
from zeta.nn.attention.flash_attention import FlashAttention

q = torch.randn(2, 4, 6, 8)
k = torch.randn(2, 4, 10, 8)
v = torch.randn(2, 4, 10, 8)

attention = FlashAttention(causal=False, dropout=0.1, flash=False)
output = attention(q, k, v)

print(output.shape)
