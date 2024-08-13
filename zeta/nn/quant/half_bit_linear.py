import torch
from torch import Tensor, nn


class HalfBitLinear(nn.Module):
    """
    A custom linear layer with half-bit quantization.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        weight (torch.Tensor): Learnable weight parameters of the layer.
        bias (torch.Tensor): Learnable bias parameters of the layer.

    Examples:
    # Example usage
    in_features = 256
    out_features = 128
    model = HalfBitLinear(in_features, out_features)
    input_tensor = torch.randn(1, in_features)
    output = model(input_tensor)
    print(output)

    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the half-bit linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the half-bit linear transformation.
        """
        # Normalize the absolute weights to be in the range [0, 1]
        normalized_abs_weights = (
            torch.abs(self.weight) / torch.abs(self.weight).max()
        )

        # Stochastic quantization
        quantized_weights = torch.where(
            self.weight > 0,
            torch.ones_like(self.weight),
            torch.zeros_like(self.weight),
        )
        stochastic_mask = torch.bernoulli(normalized_abs_weights).to(x.device)
        quantized_weights = quantized_weights * stochastic_mask

        return nn.functional.linear(x, quantized_weights, self.bias)
