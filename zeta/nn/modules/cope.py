import torch
from torch import nn, Tensor


class CoPE(nn.Module):
    def __init__(self, npos_max: int, dim: int = None):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.parameter.Parameter(torch.zeros(1, dim, npos_max))

    def forward(self, query: Tensor, attn_logits: Tensor) -> Tensor:
        # compute positions
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)


# x = torch.randn(1, 5, 10)
# attn_logits = torch.randn(1, 5, 10)

# cope = CoPE(5, 10)
# out = cope(x, attn_logits)
# print(out)
