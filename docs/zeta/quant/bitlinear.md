# BitLinear Module Documentation
==============================

## Overview
--------

The `BitLinear` module is a custom implementation of a linear layer in a neural network, with the added functionality of bit quantization. This module is designed to work with PyTorch's `nn.Module` and can be integrated into any PyTorch model architecture.

The `BitLinear` module performs linear transformation on the input data, followed by quantization and dequantization. The quantization process is performed using the `absmax_quantize` function, which quantizes the input tensor based on the absolute maximum value.

## absmax_quantize Function
------------------------

The `absmax_quantize` function is a helper function used by the `BitLinear` module to perform quantization and dequantization of the input tensor.

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| x | torch.Tensor | The input tensor to be quantized. |
| bits | int (optional) | The number of bits to use for quantization. Default is 8. |

### Returns

| Return Value | Type | Description |
| --- | --- | --- |
| quant | torch.Tensor | The quantized tensor. |
| dequant | torch.Tensor | The dequantized tensor. |

BitLinear Class
---------------

The `BitLinear` class is a custom implementation of a linear layer that performs bit quantization on the input data.

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| in_features | int | The number of input features. |
| out_features | int | The number of output features. |
| groups | int (optional) | The number of groups for group normalization. Default is 1. |

### Methods

#### `__init__(self, in_features, out_features, groups=1)`

The constructor for the `BitLinear` class. Initializes the weight parameter and resets it.

#### `reset_parameters(self)`

Resets the weight parameter using the Kaiming uniform initialization method.

#### `forward(self, input)`

Performs the forward pass of the `BitLinear` module.

### Usage Examples

#### Example 1: Basic Usage

```python
import torch

from zeta.nn.quant import BitLinear

# Initialize the BitLinear module
linear = BitLinear(10, 20)

# Create a random tensor of size (128, 10)
input = torch.randn(128, 10)

# Perform the forward pass
output = linear(input)

# Print the size of the output
print(output.size())  # torch.Size([128, 20])
```


#### Example 2: Using Different Number of Groups

```python
import torch

from zeta.nn.quant import BitLinear

# Initialize the BitLinear module with 2 groups
linear = BitLinear(10, 20, groups=2)

# Create a random tensor of size (128, 10)
input = torch.randn(128, 10)

# Perform the forward pass
output = linear(input)

# Print the size of the output
print(output.size())  # torch.Size([128, 20])
```

#### Example 3: Integrating with a PyTorch Model

```python
import torch
from torch import nn

from zeta.nn.quant import BitLinear


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = BitLinear(10, 20)

    def forward(self, x):
        return self.linear(x)


# Initialize the model
model = MyModel()

# Create a random tensor of size (128, 10)
input = torch.randn(128, 10)

# Perform the forward pass
output = model(input)

# Print the size of the output
print(output.size())  # torch.Size([128, 20])
```


# Conclusion
----------

The `BitLinear` module provides a unique way to perform linear transformation with bit quantization. This can be particularly useful in scenarios where memory efficiency is crucial. As with any other PyTorch module, it can be easily integrated into any model architecture.