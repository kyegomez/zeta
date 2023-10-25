import torch
from einops import rearrange


def batched_dot_product(a, b):
    return rearrange(a * b, "b d -> b (d)").sum(dim=-1)


# x = torch.rand(1, 3)
# model = batched_dot_product(x, x)
# print(model.shape)
