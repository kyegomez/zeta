
import math
import torch
import torch.nn as nn
from zeta import Muon  # Assuming muon.py contains our implementation

# Simple transformer layer
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Simple attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.output(out)

# Create model
model = SimpleTransformer()

# Separate parameters for different optimizers
muon_params = []
other_params = []

for name, param in model.named_parameters():
    if any(x in name for x in ['query', 'key', 'value']):
        muon_params.append(param)
    else:
        other_params.append(param)

# Create optimizers
muon_opt = Muon(muon_params, lr=0.001)
adam_opt = torch.optim.AdamW(other_params, lr=0.001)

# Training loop example
batch_size, seq_len, d_model = 32, 16, 256
x = torch.randn(batch_size, seq_len, d_model)
target = torch.randn(batch_size, seq_len, d_model)

for step in range(10):
    # Zero gradients
    muon_opt.zero_grad()
    adam_opt.zero_grad()

    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, target)

    # Backward pass
    loss.backward()

    # Update parameters
    muon_opt.step()
    adam_opt.step()

    print(f"Step {step}, Loss: {loss.item():.4f}")
