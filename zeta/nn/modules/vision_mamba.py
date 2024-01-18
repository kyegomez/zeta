from einops import rearrange
import torch
from torch import nn
from zeta.nn.modules.ssm import SSM


class VisionMambaBlock(nn.Module):
    """
    VisionMambaBlock is a module that implements the Mamba block from the paper
    Vision Mamba: Efficient Visual Representation Learning with Bidirectional
    State Space Model


    x = torch.randn(1, 512, 512)
    model = VisionMambaBlock(
        dim=512, n_heads=8, dt_rank=64, dim_inner=512, d_state=128
    )
    output = model(x)
    print(output.shape)


    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim, n_heads, dt_rank, dim_inner, d_state):
        super().__init__()
        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

    # def forward(self, x):
    #     # x is of shape [batch_size, seq_len, dim]
    #     # Use einops to rearrange for Conv1d
    #     x_rearranged = rearrange(x, "b s d -> b d s")

    #     # Forward Conv1d
    #     forward_conv_output = self.forward_conv1d(x_rearranged)
    #     forward_conv_output = rearrange(forward_conv_output, "b d s -> b s d")

    #     # Skip Connection
    #     x = x + forward_conv_output
    #     x = self.norm(x)

    #     # Self-Attention
    #     self_attention_output = self.ssm(x)

    #     # Skip Connection
    #     x = x + self_attention_output
    #     x = self.norm(x)

    #     # Backward Conv1d
    #     x_rearranged = rearrange(x, "b s d -> b d s")
    #     backward_conv_output = self.backward_conv1d(x_rearranged)
    #     backward_conv_output = rearrange(backward_conv_output, "b d s -> b s d")

    #     # Skip Connection
    #     x = x + backward_conv_output
    #     x = self.norm(x)

    #     # Activation
    #     x = self.activation(x)

    #     return x
    def forward(self, x: torch.Tensor):
        """Forward pass of the VisionMambaBlock module.

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        # x is of shape [batch_size, seq_len, dim]
        # Use einops to rearrange for Conv1d
        skip = x
        x = self.norm(x)

        z1 = x
        x1 = x

        # forward con1d
        x1_rearranged = rearrange(x1, "b s d -> b d s")
        forward_conv_output = self.forward_conv1d(x1_rearranged)
        forward_conv_output = rearrange(forward_conv_output, "b d s -> b s d")
        x1_ssm = self.ssm(forward_conv_output)

        # backward conv x2
        x2_rearranged = rearrange(x1, "b s d -> b d s")
        x2 = self.backward_conv1d(x2_rearranged)
        x2 = rearrange(x2, "b d s -> b s d")

        # Backward ssm
        x2 = self.ssm(x2)

        # Activation
        z = self.activation(z1)

        # matmul with z + backward ssm
        x2 = x2 @ z

        # Matmul with z and x1
        x1 = x1_ssm @ z

        # Add both matmuls
        x = x1 + x2

        # Add skip connection
        return x + skip
