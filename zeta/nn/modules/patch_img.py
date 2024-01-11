import torch 
from torch import nn, Tensor, einsum
from einops import rearrange

def patch_img(x: Tensor, patches: int):
    return rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patches, p2=patches)
    
    
# x = torch.randn(2, 3, 32, 32)
# x = patch_img(x, 4)
# print(x.shape)