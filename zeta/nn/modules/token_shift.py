import torch
import torch.nn.functional as F


def token_shift(t):
    t, t_shift = t.chunk(2, dim=1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1))
    return torch.cat((t, t_shift), dim=-1)
