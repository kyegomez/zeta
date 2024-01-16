from torch import Tensor
from einops import rearrange


def patch_img(x: Tensor, patches: int):
    return rearrange(
        x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patches, p2=patches
    )
