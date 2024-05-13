import torch 
from torch import nn, Tensor


class LinearizedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        seqlen: int = 10000,
        groups: int = 1,
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
        
        # Projection
        self.proj = nn.Linear(dim, dim)
        
        # RELU
        self.act = nn.ReLU()
        
        # Groupnorm
        self.norm = nn.GroupNorm(groups, dim)
        
    def forward(self, x: Tensor) -> Tensor:
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
        return nn.GroupNorm(self.groups, s)(q_acted + k_acted + v)
        
    
        
# x = torch.randn(1, 100, 512)
# model = LinearizedAttention(512, 8)
# print(model(x))
# # torch.Size([1, 100, 512])