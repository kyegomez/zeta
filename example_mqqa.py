import torch
from zeta.nn.attention.mgqa import MGQA

# Initialize the MGQA model
model = MGQA(
    dim=512,
    n_layers=6,
    head_dim=64,
    hidden_dim=2048,
    n_heads=8,
    n_kv_heads=8,
    sliding_window=512,
    norm_eps=1e-5,
    vocab_size=30522,
    max_batch_size=0,
    attn_dropout=0.1,
    flash=True
)

# Create random inputs
x = torch.randn(10, 512)  # batch size of 10, sequence length of 512

# Forward pass
output = model(x,)

print(output.shape)  # should be the same shape as x