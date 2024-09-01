import torch
import torch.nn as nn
from einops import rearrange
from zeta.nn.modules.swiglu import SwiGLU


class CogVLMTwoAdapter(nn.Module):
    """
    CogVLMTwoAdapter module that reduces the sequence length of ViT outputs and aligns the features
    with linguistic representations using a 1D convolutional layer followed by a SwiGLU module.
    """

    def __init__(self, input_dim: int):
        """
        Initialize the CogVLMTwoAdapter module.

        Args:
            input_dim (int): The dimension of the input features.
        """
        super(CogVLMTwoAdapter, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=2,
            stride=2,
        )
        self.swiglu = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CogVLMTwoAdapter module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: The output tensor after applying the 1D convolution and SwiGLU module.
        """
        # Rearrange input tensor to match the expected input shape for Conv1d (batch, input_dim, sequence_length)
        x = rearrange(x, "b s d -> b d s")

        # Apply the convolution
        x = self.conv(x)

        # Rearrange back to (batch, sequence_length, input_dim)
        x = rearrange(x, "b d s -> b s d")

        # Apply SwiGLU module
        x = self.swiglu(x)

        return x


# # Example usage
# if __name__ == "__main__":
#     # Example input (batch, sequence_length, dimension)
#     x = torch.randn(2, 4, 3)  # Adjust these dimensions as needed
#     model = CogVLMTwoAdapter(input_dim=3)
#     print(model(x).shape)
