import torch
from torch import nn
from torch import abs, softmax, sqrt, tensor, topk


class SparQAttention(nn.Module):
    """
    Sparse and Quantized Attention (SparQAttention) is a novel attention mechanism
    that approximates the attention scores using the r largest components of the query matrix
    and then gathers the top k positions based on the approximate attention scores.


    Methods:
        forward(Q, K, V, V_mean, M, r, k): Computes the Sparse and Quantized attention.

    Examples:
    >>> import torch
    >>> from zeta.nn.modules import SparQAttention
    >>> attention = SparQAttention()
    >>> batch_size, heads, seq_length, dim = 2, 4, 10, 64
    >>> Q = torch.randn(batch_size, heads, seq_length, dim)
    >>> K = torch.randn(batch_size, heads, seq_length, dim)
    >>> V = torch.randn(batch_size, heads, seq_length, dim)
    >>> V_mean = torch.randn(batch_size, heads, 1, dim)
    >>> M = torch.randn(batch_size, heads, seq_length, seq_length)
    >>> r = 5  # Number of largest components for approximation
    >>> k = 5  # Number of top positions for attention
    >>> output = attention.forward(Q, K, V, V_mean, M, r, k)
    >>> print(output)




    """

    def __init__(self, dim: int = None, heads: int = None, *args, **kwargs):
        """Initialize the SparQAttention class."""
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.heads = heads

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        V_mean: torch.Tensor,
        M: torch.Tensor,
        r: int,
        k: int,
        *args,
        **kwargs,
    ):
        """
        Computes the Sparse and Quantized attention.

        Args:
            Q (Tensor): Query matrix.
            K (Tensor): Key matrix.
            V (Tensor): Value matrix.
            V_mean (Tensor): Mean of values.
            M (Tensor): Mask.
            r (int): Number of largest components for approximation.
            k (int): Number of top positions for attention.

        Returns:
            Tensor: The result of applying sparse quantized attention.
        """
        try:
            # # Make sure that the input tensors match the specified dimensions
            # assert Q.size(1) == self.heads and Q.size(-1) == self.dim, \
            #     "Query tensor dimensions do not match the specified number of heads and head dimension"
            # assert K.size(1) == self.heads and K.size(-1) == self.dim, \
            #     "Key tensor dimensions do not match the specified number of heads and head dimension"
            # assert V.size(1) == self.heads and V.size(-1) == self.dim, \
            #     "Value tensor dimensions do not match the specified number of heads and head dimension"

            # Gather function
            def gather(t, dim, i):
                dim += (dim < 0) * t.dim()
                return t.gather(
                    dim,
                    i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]),
                )

            # Attention function
            def attn(q, k, v, m):
                s = q @ k.transpose(-1, -2) / sqrt(tensor(q.shape[-1])) + m
                return softmax(s, dim=-1) @ v

            # 1. Approximate attention scores using r largest components of Q
            i1 = topk(abs(Q), r, -1).indices
            Q_hat, K_hat = gather(Q, -1, i1), gather(K, -1, i1)
            scale = sqrt(
                Q.shape[-1]
                * abs(Q_hat).sum(dim=-1, keepdim=True)
                / abs(Q).sum(dim=-1, keepdim=True)
            )
            s_hat = softmax(Q_hat @ K_hat.transpose(-1, -2) / scale + M, dim=-1)

            # 2. Gather top k positions based on approximate attention scores & run attention
            i2 = topk(s_hat, k, -1).indices
            iKV = i2[..., 0, :, None]
            K, V, M = gather(K, -2, iKV), gather(V, -2, iKV), gather(M, -1, i2)
            y_ = attn(Q, K, V, M)

            # 3. Estimate the total score of the top k, and interpolate with V_mean
            alpha = gather(s_hat, -1, i2).sum(-1, keepdim=True)
            return alpha * y_ + (1 - alpha) * V_mean
        except Exception as e:
            raise ValueError(f"Error in SPARQ attention computation: {e}")


# Example usage
num_heads = 4
head_dim = 64
attention = SparQAttention(num_heads, head_dim)

# Generate random tensors with the specified dimensions
batch_size, seq_length = 2, 10
Q = torch.randn(batch_size, num_heads, seq_length, head_dim)
K = torch.randn(batch_size, num_heads, seq_length, head_dim)
V = torch.randn(batch_size, num_heads, seq_length, head_dim)
V_mean = torch.randn(batch_size, num_heads, 1, head_dim)
M = torch.randn(batch_size, num_heads, seq_length, seq_length)

# Compute the Sparse and Quantized attention
r = 5  # Number of largest components for approximation
k = 5  # Number of top positions for attention
output = attention.forward(Q, K, V, V_mean, M, r, k)

# Output tensor
print(output)
