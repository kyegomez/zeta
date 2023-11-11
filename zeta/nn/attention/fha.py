"""
Does not work yet


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FMA(nn.Module):
    """
    Fast Multipole Attention (FMA) Module.
    Implements a hierarchical attention mechanism with downsampling for efficiency.
    """

    def __init__(
        self, d_model, n_heads=1, group_size=2, approximation_rank=1, max_seq_length=32
    ):
        """
        Initialize the FMA module.
        :param d_model: Dimension of the model.
        :param n_heads: Number of attention heads.
        :param group_size: Size of groups at the finest level.
        :param approximation_rank: Rank of approximation for off-diagonal blocks.
        :param max_seq_length: Maximum sequence length to support.
        """
        super(FMA, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.group_size = group_size
        self.approximation_rank = approximation_rank
        self.depth = int(math.log2(d_model / group_size)) - 1

        # Adjust convolution layers based on maximum sequence length
        self.key_convs = nn.ModuleList()
        self.value_convs = nn.ModuleList()

        for i in range(1, self.depth + 1):
            kernel_size = min(2**i * group_size, max_seq_length)
            stride = kernel_size
            self.key_convs.append(
                nn.Conv1d(d_model, d_model, kernel_size, stride, groups=d_model)
            )
            self.value_convs.append(
                nn.Conv1d(d_model, d_model, kernel_size, stride, groups=d_model)
            )

        # Linear layers for queries, keys, and values
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Forward pass for FMA.
        :param x: Input sequence of shape (batch_size, seq_length, d_model).
        :return: Output sequence.
        """
        batch_size, seq_length, _ = x.size()

        # Compute queries, keys, and values
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)

        # Downsample keys and values
        Ks = [K]
        Vs = [V]
        for key_conv, value_conv in zip(self.key_convs, self.value_convs):
            Ks.append(key_conv(K.transpose(1, 2)).transpose(1, 2))
            Vs.append(value_conv(V.transpose(1, 2)).transpose(1, 2))

        # Compute attention scores and outputs at each level
        attention_output = torch.zeros_like(x)
        for level in range(self.depth + 1):
            Qi = Q if level == 0 else self.downsample(Q, level)
            Ki = Ks[level]
            Vi = Vs[level]

            # Compute attention scores
            attention_scores = torch.bmm(Qi, Ki.transpose(1, 2)) / math.sqrt(
                self.d_model
            )
            attention_scores = F.softmax(attention_scores, dim=-1)

            # Compute attention output
            attention_output += torch.bmm(attention_scores, Vi)

        return attention_output

    def downsample(self, x, level):
        """
        Downsample the input sequence for a given level.
        :param x: Input sequence.
        :param level: Level of downsampling.
        :return: Downsampled sequence.
        """
        stride = 2 ** (level - 1) * self.group_size
        return F.avg_pool1d(
            x.transpose(1, 2), kernel_size=stride, stride=stride
        ).transpose(1, 2)


# Example usage
seq_length = 32  # Example sequence length
d_model = 512  # Example dimension of the model
x = torch.randn(1, seq_length, d_model)  # Example input

fma = FMA(d_model)
output = fma(x)

print(output.shape)  # Expected output shape: [1, seq_length, d_model]
