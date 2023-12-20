import torch
from torch import nn
from zeta.nn.modules.rms_norm import RMSNorm
from zeta.nn.modules.residual import Residual


class Mamba(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        bias: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                Residual(self.rmsnorm, nn.Linear(dim, dim, bias=bias))
                for _ in range(depth)
            ]
        )
        self.rmsnorm = RMSNorm(dim)
        self.linear = nn.Linear(dim, vocab_size, bias=bias)
        self.linear.weight = self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.rmsnorm(x)
        logits = self.linear(x)

        return logits


# class MambaBlock(nn.Module):
#     def __init__(
#         self,
#         dim,
#         inner_dim,
#         bias: bool = False,
#         conv_bias=None,
#         dim_conv=None,
#         *args,
#         **kwargs,
#     ):
#         super().__init__()
