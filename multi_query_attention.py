import torch
from zeta import MultiQueryAttention

# Model
model = MultiQueryAttention(
    dim=512,
    heads=8,
)

# Input
text = torch.randn(2, 4, 512)

# Output
output, _, _ = model(text)
print(output.shape)
print(output)
