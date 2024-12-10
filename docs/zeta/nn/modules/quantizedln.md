# Module/Class Name: QuantizedLN

## Overview
`QuantizedLN` is a PyTorch module built on the lower-level `nn.Module` class. This module is designed for applying a form of normalization where the layer inputs are transformed to have zero mean and one standard deviation, and subsequently quantized. The main purpose of this module is to provide normalized inputs with reduced precision for performance and memory optimization purposes, seen typically in low-resource environments like mobile devices.

The 'LN' in the class name refers to Layer Normalization, a technique that normalizes the inputs across the features instead of the batch size. The 'Quantized' in the class name signifies that the normalized output is then quantized to a specified bit size for memory and speed optimizations.

```python
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
  """
    ...

  def forward(self, x: Tensor):
  """
  Forward pass of the QuantizedLN module.

  Args:
      x (torch.Tensor): Input tensor.

  Returns:
      torch.Tensor: Output tensor after applying quantization and layer normalization.
  """
    ...
```

## Parameters
The `QuantizedLN` class takes the following arguments during initialization:

| Parameter Name | Type | Description | Default Value |
| --- | --- | --- | --- |
| normalized_shape | int or tuple | The expected input shape | Required |
| bits | int | Number of bits for quantization | 8 |
| eps | float | A small value added to the denominator for numerical stability | 1e-5 |
| element_wise_affine | bool | If True, includes learnable affine parameters | True |

## Methods
The `QuantizedLN` class has the following methods:

| Method Name | Args | Returns | Description |
| --- | --- | --- | --- |
| init | normalized_shape, bits, eps, element_wise_affine | None | Initializes the QuantizedLN module |
| forward | x | torch.Tensor | Performs the forward pass |

## Usage Examples

Below are three examples of how to use the `QuantizedLN` module.

### Example 1

```python
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from zeta.nn.modules import QuantizedLN

# Define input tensor
x = torch.randn(128, 10)
# Create module instance
ln = QuantizedLN(10)
# Apply module to input
output = ln(x)
```

### Example 2

Define a custom network that uses have the `QuantizedLN` module:

```python
import torch.nn as nn

from zeta.nn.modules import QuantizedLN


class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(128, 256)
        self.ln = QuantizedLN(256)

    def forward(self, x):
        x = self.layer1(x)
        x = self.ln(x)
        return x


# Define input tensor
x = torch.randn(128, 10)

# Create network instance
network = CustomNetwork()

# Forward pass
output = network(x)
```

### Example 3

The `QuantizedLN` module in a multi-layer setup:

```python
import torch.nn as nn

from zeta.nn.modules import QuantizedLN


class DeepNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(128, 256)
        self.ln1 = QuantizedLN(256)
        self.layer2 = nn.Linear(256, 512)
        self.ln2 = QuantizedLN(512)

    def forward(self, x):
        x = self.layer1(x)
        x = self.ln1(x)
        x = self.layer2(x)
        x = self.ln2(x)
        return x


# Define input tensor
x = torch.randn(128, 10)

# Create network instance
network = DeepNetwork()

# Forward pass
output = network(x)
```

## Additional Notes:

Please make sure that the `absmax_quantize` function used in the `forward` method is properly defined in the scope of this class or is imported correctly from an external module. It is a quantization function that is not included by default in PyTorch's `nn` module. Failure to define or import this function will result in errors during execution.
