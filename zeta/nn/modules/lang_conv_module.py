import torch
from torch import nn


class ConvolutionLanguageBlock(nn.Module):
    """
    Convolutional block for language modeling.
    --------------------------------------------
    A convolutional block that consists of multiple 1D convolutional layers,
    optional batch normalization, dropout, and a flexible choice of activation functions.
    This block is designed to maintain the input's dimensionality through the network,
    making it suitable for tasks that require consistent input and output dimensions.

    Parameters:
    - in_channels (int): Number of channels in the input tensor.
    - out_channels (int): Number of channels produced by the convolution.
    - kernel_size (int): Size of the convolving kernel.
    - num_layers (int, optional): Number of convolutional layers. Default: 1
    - stride (int, optional): Stride of the convolution. Default: 1
    - padding (int, optional): Zero-padding added to both sides of the input. Default: 1
    - dilation (int, optional): Spacing between kernel elements. Default: 1
    - activation (str, optional): Type of activation function. Options: 'relu', 'gelu'. Default: 'relu'
    - use_batchnorm (bool, optional): If True, includes batch normalization. Default: False
    - dropout (float, optional): Dropout rate. Default: 0.0

    Examples:
        >>> import torch
        >>> from attnconv.main import ConvolutionLanguageBlock
        >>> x = torch.randn(1, 512, 1024)
        >>> block = ConvolutionLanguageBlock(512, 512, 3, 1, 1, 1)
        >>> out = block(x)
        >>> out.shape
        torch.Size([1, 512, 1024])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        depth=1,
        stride=1,
        activation="gelu",
        batchnorm=False,
        dilation=1,
        dropout=0.1,
    ):
        super(ConvolutionLanguageBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.depth = depth
        self.stride = stride
        self.activation = activation
        self.batchnorm = batchnorm
        self.dilation = dilation

        layers = []
        for _ in range(depth):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            )
            if batchnorm:
                layers.append(nn.BatchNorm1d(out_channels))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels  # For stacking layers

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass with residual connection.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Apply residual connection if dimensions match
        residual = x if x.size(1) == self.conv_layers[0].in_channels else None

        # Apply convolutional layers
        x = self.conv_layers(x)

        # Apply residual connection
        if residual is not None:
            x = x + residual

        # Return output
        return x
