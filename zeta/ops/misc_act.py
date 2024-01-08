from torch import nn, Tensor
import torch.nn.functional as F



# These extra constant values ensure that the activations
# are variance preserving
class VPGELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input) * 1.7015043497085571


class VPReLU(nn.Module):
    """
    Variational Parametric Rectified Linear Unit (VPReLU) activation function.

    Args:
        inplace (bool, optional): If set to True, will modify the input tensor in-place. Default is False.

    Attributes:
        inplace (bool): Flag indicating whether the input tensor is modified in-place.

    """

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(VPReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of the VPReLU activation function.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the VPReLU activation function.

        """
        return F.relu(input, inplace=self.inplace) * 1.7139588594436646

    def extra_repr(self) -> str:
        """
        Extra representation of the VPReLU module.

        Returns:
            str: Extra representation string.

        """
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str