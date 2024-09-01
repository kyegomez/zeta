from torch import Tensor, nn


class FusedProjSoftmax(nn.Module):
    """
    FusedProjSoftmax is a module that applies a linear projection followed by a softmax operation.

    Args:
        dim (int): The input dimension.
        dim_out (int): The output dimension.
        dim_axis (int, optional): The axis along which the softmax operation is applied. Defaults to -1.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        proj (nn.Linear): The linear projection layer.
        softmax (nn.Softmax): The softmax operation layer.

    Examples:
        x = torch.rand(1, 2, 3)
        model = FusedProjSoftmax(3, 4)
        out = model(x)
        print(out.shape)
    """

    def __init__(
        self, dim: int, dim_out: int, dim_axis: int = -1, *args, **kwargs
    ):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out, *args, **kwargs)
        self.softmax = nn.Softmax(dim=dim_axis)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the FusedProjSoftmax module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying linear projection and softmax.
        """
        return self.softmax(self.proj(x))
