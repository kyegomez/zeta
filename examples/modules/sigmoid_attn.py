import torch
from zeta import SigmoidAttention
from loguru import logger

batch_size = 32
seq_len = 128
dim = 512
heads = 8

x = torch.rand(batch_size, seq_len, dim)
mask = torch.ones(batch_size, seq_len, seq_len)  # Example mask

sigmoid_attn = SigmoidAttention(dim, heads, seq_len)
output = sigmoid_attn(x, mask)
logger.info(f"Final output shape: {output.shape}")
