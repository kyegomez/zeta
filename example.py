"""
This script demonstrates the usage of the FlashAttention module from zeta.nn.
"""

import torch
from zeta.nn import FlashAttention

# Set random seed for reproducibility
torch.manual_seed(42)

# Define input tensor shapes
batch_size, num_heads, seq_len_q, d_head = 2, 4, 6, 8
seq_len_kv = 10

# Create random input tensors
q = torch.randn(batch_size, num_heads, seq_len_q, d_head)
k = torch.randn(batch_size, num_heads, seq_len_kv, d_head)
v = torch.randn(batch_size, num_heads, seq_len_kv, d_head)

# Initialize FlashAttention module
attention = FlashAttention(causal=False, dropout=0.1, flash=False)
print("FlashAttention configuration:", attention)

# Perform attention operation
output = attention(q, k, v)

print(f"Output shape: {output.shape}")

# Optional: Add assertion to check expected output shape
assert output.shape == (batch_size, num_heads, seq_len_q, d_head), "Unexpected output shape"