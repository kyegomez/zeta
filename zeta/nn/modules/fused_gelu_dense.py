import torch
from torch import nn


class FusedDenseGELUDense(nn.Module):
    """FuseFusedDenseGELUDense

    Args
        dim (int): Input dimension
        dim_out (int): Output dimension
        bias (bool, optional): Bias. Defaults to True.
        has_fp16_weights (bool, optional): Use fp16 weights. Defaults to False.
        threshold (float, optional): Threshold for quantization. Defaults to 6.0.

    Examples:
        >>> x = torch.randn(1, 512)
        >>> model = FusedDenseGELUDense(512, 1024)
        >>> out = model(x)
        >>> out.shape
        torch.Size([1, 512])
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        bias: bool = True,
        has_fp16_weights: bool = False,
        threshold: float = 6.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.bias = bias
        self.has_fp16_weights = has_fp16_weights
        self.threshold = threshold

        try:
            import bitsandbytes as bnb

            # Using bitsandbytes for quantization
            self.dense1 = bnb.nn.Linear8bitLt(
                dim,
                dim_out,
                bias=bias,
                has_fp16_weights=has_fp16_weights,
                threshold=threshold,
                *args,
                **kwargs,
            )

            # Reverse
            self.dense2 = bnb.nn.Linear8bitLt(
                dim_out,
                dim,
                bias=bias,
                has_fp16_weights=has_fp16_weights,
                threshold=threshold,
                *args,
                **kwargs,
            )

        except ModuleNotFoundError:
            # Using torch.nn.Linear
            self.dense1 = nn.Linear(dim, dim_out, bias=bias * args, **kwargs)

            # Dense 2
            self.dense2 = nn.Linear(dim_out, dim, bias=bias * args, **kwargs)

        # Activation
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): x input

        Returns:
            torch.Tensor: _description_
        """
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)
        return x
