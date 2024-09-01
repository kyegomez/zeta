from torch import Tensor, nn

from zeta.nn.quant.bitlinear import absmax_quantize


class QuantizedLN(nn.Module):
    def __init__(
        self,
        normalized_shape,
        bits: int = 8,
        eps=1e-5,
        element_wise_affine=True,
    ):
        """
        Initializes a QuantizedLN module.

        Args:
            normalized_shape (int or tuple): The expected input shape.
            bits (int, optional): Number of bits for quantization. Defaults to 8.
            eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-5.
            element_wise_affine (bool, optional): Whether to include learnable affine parameters. Defaults to True.

        Examples::
        x = torch.randn(128, 10)
        ln = QuantizedLN(10)
        output = ln(x)
        print(output)

        """
        super().__init__()
        self.bits = bits
        self.ln = nn.LayerNorm(
            normalized_shape, eps=eps, elementwise_affine=element_wise_affine
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the QuantizedLN module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying quantization and layer normalization.
        """
        _, x_dequant = absmax_quantize(x, bits=self.bits)
        return self.ln(x_dequant)
