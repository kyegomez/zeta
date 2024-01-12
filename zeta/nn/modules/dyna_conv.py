import torch
import torch.nn as nn
import torch.nn.functional as F


class DynaConv(nn.Module):
    """
    DynaConv is an experimental replacement for traditional convolutional layers.

    Instead of using fixed filters, this layer dynamically generates convolutional
    kernels based on the input features using a small neural network.

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
        super(DynaConv, self).__init__()
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
        # Correctly calculate the gain for kaiming_uniform
        gain = nn.init.calculate_gain(
            "tanh"
        )  # since we are using Tanh in the kernel generator
        # Initialize the weights of the kernel generator network
        nn.init.kaiming_uniform_(self.kernel_generator[0].weight, a=gain)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.kernel_generator[0].weight
            )
            bound = 1 / torch.sqrt(
                torch.tensor(fan_in, dtype=torch.float32)
            )  # Convert fan_in to a tensor before sqrt
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Forward pass of the DynaConv layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The result of the dynamic convolution.
        """
        batch_size, _, H, W = x.shape
        # Generate dynamic kernels
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        kernels = self.kernel_generator(x_unfold.transpose(1, 2)).view(
            batch_size, self.out_channels, -1
        )

        # Perform convolution with dynamic kernels
        output = kernels.bmm(x_unfold).view(batch_size, self.out_channels, H, W)

        # Add bias if necessary
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output


# Example usage:
dynaconv = DynaConv(
    in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
)
input_tensor = torch.randn(1, 3, 224, 224)  # Example input batch
output = dynaconv(input_tensor)
