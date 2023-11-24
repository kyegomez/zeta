from torch import nn
from einops import rearrange


class EinopsToAndFrom(nn.Module):
    """
    EinopsToAndFrom module for converting between einops patterns.

    This module is useful for converting between einops patterns in a
    differentiable manner. It is designed to be used in conjunction with
    einops_poly.py.

    Attributes:
        from_pattern (str): The input einops pattern.
        to_pattern (str): The output einops pattern.

    Usage:
        - Instantiate the module and pass a tensor to it.

    Example:
        >>> x = torch.randn(1, 2, 3, 4)
        >>> print(x.shape)
        torch.Size([1, 2, 3, 4])
        >>> module = EinopsToAndFrom("b c h w", "b h w c")
        >>> y = module(x)
        >>> print(y.shape)
        torch.Size([1, 3, 4, 2])

    """

    def __init__(self, from_pattern, to_pattern):
        super().__init__()
        self.from_pattern = from_pattern
        self.to_pattern = to_pattern
        self.fn = FileNotFoundError

        if "..." in from_pattern:
            before, after = [
                part.strip().split() for part in from_pattern.split("...")
            ]
            self.reconsitute_keys = tuple(
                zip(before, range(len(before)))
            ) + tuple(zip(after, range(-len(after), 0)))
        else:
            split = from_pattern.strip().split()
            self.reconsitute_keys = tuple(zip(split, range(len(split))))

    def forward(self, x, **kwargs):
        """
        forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.


        """
        shape = x.shape
        reconsitute_kwargs = {
            key: shape[position] for key, position in self.reconsitute_keys
        }
        x = rearrange(x, f"{self.from_pattern} -> {self.to_pattern}")
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f"{self.to_pattern} -> {self.from_pattern}", **reconsitute_kwargs
        )
        return x
