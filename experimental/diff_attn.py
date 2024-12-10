import torch
import torch.nn.functional as F
from torch import nn, Tensor
from math import sqrt
from loguru import logger


class DiffAttn(nn.Module):
    """
    Differentiated Attention module that applies attention by comparing two sets of queries (Q1, Q2)
    with their corresponding keys (K1, K2), modulated by a regularization parameter λ.

    Args:
        X (Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
        W_q (Tensor): Query weight matrix of shape [input_dim, 2 * d].
        W_k (Tensor): Key weight matrix of shape [input_dim, 2 * d].
        W_v (Tensor): Value weight matrix of shape [input_dim, 2 * d].
        λ (float): Regularization parameter to control the difference between softmax(A1) and softmax(A2).

    Returns:
        Tensor: Output tensor after applying differentiated attention.
    """

    def __init__(self, d: int):
        super(DiffAttn, self).__init__()
        self.d = d

    def forward(
        self, X: Tensor, W_q: Tensor, W_k: Tensor, W_v: Tensor, λ: float
    ) -> Tensor:
        logger.info("Executing DiffAttn forward pass")

        # Compute the Queries, Keys, and Values
        Q1, Q2 = self.split(X @ W_q)
        K1, K2 = self.split(X @ W_k)
        V = X @ W_v

        s = 1 / sqrt(self.d)

        # Attention scores
        A1 = (Q1 @ K1.transpose(-1, -2)) * s
        A2 = (Q2 @ K2.transpose(-1, -2)) * s

        # Apply softmax to attention scores
        A1_softmax = F.softmax(A1, dim=-1)
        A2_softmax = F.softmax(A2, dim=-1)

        # Return the differentiated attention
        result = (A1_softmax - λ * A2_softmax) @ V
        return result

    @staticmethod
    def split(X: Tensor) -> (Tensor, Tensor):
        """
        Splits the input tensor X into two equal halves along the last dimension.

        Args:
            X (Tensor): Input tensor to be split.

        Returns:
            Tuple[Tensor, Tensor]: Two tensors resulting from the split.
        """
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]


class MultiHead(nn.Module):
    """
    MultiHead attention mechanism that applies differentiated attention across multiple heads.

    Args:
        X (Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
        W_q (List[Tensor]): List of query weight matrices, one per head.
        W_k (List[Tensor]): List of key weight matrices, one per head.
        W_v (List[Tensor]): List of value weight matrices, one per head.
        W_o (Tensor): Output weight matrix of shape [h * 2 * d, output_dim].
        λ (float): Regularization parameter to control the difference between softmax(A1) and softmax(A2).
        λinit (float): Initialization lambda to scale the final output.

    Returns:
        Tensor: Output tensor after applying multi-head differentiated attention.
    """

    def __init__(self, h: int, d: int, λinit: float):
        super(MultiHead, self).__init__()
        self.h = h
        self.d = d
        self.λinit = λinit
        self.diff_attn_heads = nn.ModuleList([DiffAttn(d) for _ in range(h)])

    def forward(
        self,
        X: Tensor,
        W_q: Tensor,
        W_k: Tensor,
        W_v: Tensor,
        W_o: Tensor,
        λ: float,
    ) -> Tensor:
        logger.info("Executing MultiHead forward pass")

        # Apply differentiated attention for each head
        O_list = [
            self.diff_attn_heads[i](X, W_q[i], W_k[i], W_v[i], λ)
            for i in range(self.h)
        ]

        # Concatenate outputs from each head
        O_concat = torch.cat(O_list, dim=-1)

        # Reshape input for GroupNorm to [batch_size, num_channels, seq_len]
        O_concat = O_concat.permute(0, 2, 1)

        # Apply GroupNorm (adjust num_groups based on your configuration)
        O_normalized = nn.GroupNorm(
            num_groups=8, num_channels=O_concat.shape[1]
        )(O_concat)

        # Revert shape back to [batch_size, seq_len, num_channels]
        O_normalized = O_normalized.permute(0, 2, 1)

        # Apply the output transformation and scale by λinit
        result = O_normalized * (1 - self.λinit)
        result = result @ W_o

        return result


# Example usage:

# Example dimensions
batch_size, seq_len, input_dim, d, h = 32, 128, 64, 32, 8
λ, λinit = 0.1, 0.05

# Create random weight matrices
W_q = [torch.randn(input_dim, 2 * d) for _ in range(h)]
W_k = [torch.randn(input_dim, 2 * d) for _ in range(h)]
W_v = [torch.randn(input_dim, 2 * d) for _ in range(h)]
W_o = torch.randn(h * 2 * d, input_dim)

# Create random input tensor
X = torch.randn(batch_size, seq_len, input_dim)

# Instantiate and run the multi-head attention
multi_head = MultiHead(h=h, d=d, λinit=λinit)
output = multi_head(X, W_q, W_k, W_v, W_o, λ=λ)

logger.info(f"Output shape: {output.shape}")
