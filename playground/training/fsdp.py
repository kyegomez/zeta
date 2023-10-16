import torch.nn as nn
from zeta.training import fsdp

# Define your PyTorch model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Wrap the model with FSDP with custom settings (full model sharding, bf16 precision)
fsdp_model = fsdp(model, mp="bf16", shard_strat="FULL_SHARD")
