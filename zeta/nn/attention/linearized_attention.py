import torch
from torch import nn, Tensor


class LinearizedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        seqlen: int = 1000,
        groups: int = 1,
        mask_on: bool = False,
        *args,
        **kwargs,
    ):
        """
        Linearized Attention module.

        Args:
            dim (int): Dimension of the input tensor.
            heads (int): Number of attention heads.
            seqlen (int): Length of the input sequence.
            groups (int, optional): Number of groups for group normalization. Defaults to 1.
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.seqlen = seqlen
        self.groups = groups
        self.mask_on = mask_on

        # Projection
        self.proj = nn.Linear(dim, dim)

        # RELU
        self.act = nn.ReLU()

        # Groupnorm
        self.norm = nn.GroupNorm(groups, dim)

        # Mask Tensor
        self.mask_tensor = torch.zeros(1, seqlen).bool()

    def forward(self, x: Tensor, mask: bool = None) -> Tensor:
        """
        Forward pass of the LinearizedAttention module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying LinearizedAttention.
        """
        b, s, d = x.shape
        q = self.proj(x)
        k = self.proj(x)
        v = self.proj(x)

        # Projected again
        q_p = self.proj(q)
        q_k = self.proj(k)

        # Apply the relu
        q_acted = self.act(q_p)
        k_acted = self.act(q_k)

        # Groupnorm
        output = nn.GroupNorm(self.groups, s)(q_acted + k_acted + v)

        # Apply mask
        if mask is not None:
            if self.mask_on is True:
                mask = self.mask_tensor
            else:
                output = output.masked_fill(mask.unsqueeze(-1), float("-inf"))
                print(output.shape)

        return output


# x = torch.randn(1, 10, 20)
# model = LinearizedAttention(20, 8, mask_on=True)
# print(model(x))
# # torch.Size([1, 10, 20])
