from typing import List, TypeVar

import torch
from einops import rearrange

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class Tensor(torch.nn.Module):
    def __init__(
        self,
        data: torch.Tensor,
        shape: List[str],
        to: List[str],
    ):
        super().__init__()
        self.data = data
        self.shape = shape
        self.to = to

    def __call__(self):
        shape = " ".join(self.shape)
        to = "".join(self.to)

        return rearrange(
            self.data,
            shape + " -> " + to,
        )


# # Example
# x = torch.randn(2, 4, 6, 8)

# model = Tensor(
#     data=x,
#     shape=["b d s h"],
#     to=['b h s d']
# )

# out = model()
# print(out)
