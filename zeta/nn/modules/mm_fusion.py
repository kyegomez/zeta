import torch
from torch import nn
from einops import rearrange


class MultiModalFusion(nn.Module):
    def forward(self, x, y):
        return torch.einsum("bi, bj -> bij", x, y)


# # #random
# x = torch.rand(1, 3)
# y = torch.rand(1, 3)
# model = MultiModalFusion()
# out = model(x, y)
# print(out.shape)
