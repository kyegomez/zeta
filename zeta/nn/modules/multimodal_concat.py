import torch
from einops import rearrange


def multimodal_concat(*features):
    return rearrange(features, "... b d -> b ... d", merge="d")


# # #random
# x = torch.rand(1, 3, 33)
# model = multimodal_concat(x, x)
# print(model.shape)
