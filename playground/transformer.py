import torch
from zeta.nn import Transformer, Decoder

logits = torch.randint(0, 256, (1, 1024))

# Example 1: Basic Usage
transformer = Transformer(
    num_tokens=20000,
    max_seq_len=1024,
    attn_layers=Decoder(dim=512, depth=12, heads=8),
)

logits = transformer(logits)
print(logits)
