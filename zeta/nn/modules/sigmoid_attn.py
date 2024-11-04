import torch
import torch.nn as nn
import math
from loguru import logger
from typing import Optional


class SigmoidAttention(nn.Module):
    """
    Implements Sigmoid Attention Mechanism.

    This replaces the traditional softmax in attention with a sigmoid function.
    Additionally, a constant scalar bias based on the sequence length is introduced.

    Args:
        dim (int): Dimension of the model (input size).
        heads (int): Number of attention heads.
        seq_len (int): The length of the input sequence.
        dropout (float, optional): Dropout rate. Default is 0.1.
        bias (bool, optional): Whether to include bias in linear layers. Default is True.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        seq_len: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super(SigmoidAttention, self).__init__()

        logger.info(
            f"Initializing SigmoidAttention with dim={dim}, heads={heads}, seq_len={seq_len}, dropout={dropout}, bias={bias}"
        )
        self.dim = dim
        self.heads = heads
        self.seq_len = seq_len
        self.head_dim = dim // heads

        assert self.head_dim * heads == dim, "dim must be divisible by heads"
        logger.debug(f"Each attention head has {self.head_dim} dimensions.")

        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Create a constant scalar bias based on the sequence length
        self.bias = nn.Parameter(
            torch.ones(1) * math.sqrt(self.seq_len), requires_grad=False
        )
        logger.debug(
            f"Scalar bias initialized as {self.bias.item()} based on sequence length."
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Sigmoid Attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            mask (Optional[torch.Tensor], optional): Mask tensor to prevent attention to certain positions.
                                                     Should be of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        logger.info(f"Running forward pass with input shape {x.shape}")
        batch_size, seq_len, _ = x.size()

        # Linear projections for query, key, and value
        Q = (
            self.query(x)
            .view(batch_size, seq_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key(x)
            .view(batch_size, seq_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value(x)
            .view(batch_size, seq_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )

        logger.debug(f"Q, K, V shapes: {Q.shape}, {K.shape}, {V.shape}")

        # Scaled dot-product attention with sigmoid instead of softmax
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores / self.bias  # Apply the constant scalar bias
        attn = torch.sigmoid(scores)
        logger.debug(f"Attention scores computed with sigmoid: {attn.shape}")

        # Apply the mask (optional)
        if mask is not None:
            logger.debug(f"Original mask shape: {mask.shape}")
            # Expand the mask to match the attention scores shape
            mask = mask.unsqueeze(1)  # Adds dimension for heads
            logger.debug(f"Expanded mask shape: {mask.shape}")
            attn = attn.masked_fill(mask == 0, -1e9)
            logger.debug("Mask applied to attention scores.")

        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.dim)
        )

        logger.info(f"Output shape: {output.shape}")
        return self.out(output)


# # Example usage
# if __name__ == "__main__":
#     import torch
#     from zeta import SigmoidAttention
#     batch_size = 32
#     seq_len = 128
#     dim = 512
#     heads = 8

#     x = torch.rand(batch_size, seq_len, dim)
#     mask = torch.ones(batch_size, seq_len, seq_len)  # Example mask

#     sigmoid_attn = SigmoidAttention(dim, heads, seq_len)
#     output = sigmoid_attn(x, mask)
#     logger.info(f"Final output shape: {output.shape}")
