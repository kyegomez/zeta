import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QUIK(nn.Module):
    """
    QUICK Linear layer

    Args:
        in_features: number of input features
        out_features: number of output features
        bias: whether to use bias

    Returns:
        x + proj(x) where proj is a small MLP


    Usage:
    import torch

    # Initialize the QUIK module
    quik = QUIK(in_features=784, out_features=10)

    # Create some dummy data, e.g., simulating a batch of MNIST images
    data = torch.randn(10, 784)

    # Run the data through the network
    output = quik(data)
    print(output)


    """

    def __init__(self, in_features, out_features, bias=True):
        super(QUIK, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.quantize_range = (
            8  # Assuming 4-bit quantization, so range is [-8, 7]
        )
        self.half_range = self.quantize_range // 2

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def quantize(self, input_tensor):
        """
        Args:
            input_tensor: input tensor to be quantized

        Returns:
            output_tensor: quantized tensor
            zero_act: zero-point
            scale_act: scale factor



        """
        zero_act, scale_act = self.find_zero_scale(input_tensor)
        output_tensor = (
            ((input_tensor - zero_act) / scale_act - self.half_range)
            .clamp(-self.half_range, self.half_range - 1)
            .int()
        )
        return output_tensor, zero_act, scale_act

    def dequantize(self, input_tensor, zero_act, scale_act, scale_weight):
        """
        Dequantize the input tensor

        Args:
            input_tensor: input tensor to be dequantized
            zero_act: zero-point
            scale_act: scale factor
            scale_weight: scale factor for weight

        Returns:

            output_tensor: dequantized tensor


        """
        weights_reduced = self.weight.sum(dim=1)
        x = input_tensor.float() * scale_act * scale_weight
        shift = (
            zero_act + self.half_range * scale_act
        ) * weights_reduced.unsqueeze(-1)
        output_tensor = x + shift
        return output_tensor

    def find_zero_scale(self, input_tensor):
        """
        Find zero_scale


        Args:
            input_tensor: input tensor to be quantized

        Returns:
            zero_act: zero-point
            scale_act: scale factor



        """
        zero_act = input_tensor.min()
        scale_act = (input_tensor.max() - zero_act) / (2 * self.half_range)
        return zero_act, scale_act

    def forward(self, x):
        """
        Forward pass of the QUIK layer

        Args:
            x: input tensor

        Returns:
            output_tensor: output tensor after forward pass

        """
        # Quantize activations
        x_quant, zero_act, scale_act = self.quantize(x)

        # Quantized weight multiplication
        quant_weight = self.quantize(self.weight)[0]
        result = F.linear(
            x_quant.float(), quant_weight.float(), self.bias
        )  # Assuming INT32 multiplication result

        # Dequantization
        scale_weight = (self.weight.max() - self.weight.min()) / (
            2 * self.half_range
        )
        return self.dequantize(result, zero_act, scale_act, scale_weight)
