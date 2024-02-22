import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DynaConv(nn.Module):
    """
    DynaConv dynamically generates convolutional kernels based on the input features.

    This layer replaces traditional convolutional layers with a dynamic mechanism,
    where convolutional kernels are generated on-the-fly by a small neural network.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Example:
        >>> dynaconv = DynaConv(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        >>> input_tensor = torch.randn(1, 3, 224, 224) # Example input batch
        >>> output = dynaconv(input_tensor)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # The small network to generate dynamic kernels. It's a simple MLP.
        self.kernel_generator = nn.Sequential(
            nn.Linear(
                in_channels * self.kernel_size[0] * self.kernel_size[1],
                out_channels,
            ),
            nn.Tanh(),
            nn.Linear(
                out_channels,
                out_channels * self.kernel_size[0] * self.kernel_size[1],
            ),
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("tanh")
        nn.init.kaiming_uniform_(self.kernel_generator[0].weight, a=gain)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.kernel_generator[0].weight
            )
            bound = 1 / math.sqrt(
                fan_in
            )  # Use math.sqrt for the scalar square root calculation
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size, _, H, W = x.shape
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # The input to kernel_generator must match its expected input dimensions.
        # We reshape x_unfold to have dimensions [batch_size * number of patches, in_channels * kernel_size * kernel_size]
        x_unfold = rearrange(
            x_unfold,
            "b (c kh kw) l -> (b l) (c kh kw)",
            c=self.in_channels,
            kh=self.kernel_size[0],
            kw=self.kernel_size[1],
        )

        kernels = self.kernel_generator(x_unfold).view(
            batch_size,
            -1,
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        # Apply the generated kernels for each patch
        output = torch.einsum(
            "blodij,blij->bod",
            kernels,
            x_unfold.view(
                batch_size,
                -1,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            ),
        )

        # Reshape output to match the convolutional output
        output = rearrange(
            output,
            "b (h w) d -> b d h w",
            h=H // self.stride,
            w=W // self.stride,
        )

        # Add bias if necessary
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output


# # Example usage
# dynaconv = DynaConv(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
# input_tensor = torch.randn(1, 3, 224, 224)  # Example input batch
# output = dynaconv(input_tensor)
