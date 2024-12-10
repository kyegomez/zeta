# FusedDropoutLayerNorm Documentation

## Overview

The `FusedDropoutLayerNorm` module in PyTorch is designed to combine two commonly used operations in neural networks: dropout and layer normalization. This fusion aims to enhance the efficiency of the model by reducing the overhead associated with sequential operations. The module is particularly useful in scenarios where both dropout and layer normalization are critical for the model's performance.

## Class Definition

### `FusedDropoutLayerNorm`

```python
class FusedDropoutLayerNorm(nn.Module):
    """
    This class fuses Dropout and LayerNorm into a single module for efficiency.

    Args:
        dim (int): Input dimension of the layer.
        dropout (float, optional): Probability of an element to be zeroed. Defaults to 0.1.
        eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-5.
        elementwise_affine (bool, optional): A flag to enable learning of affine parameters. Defaults to True.
    """
```

## Constructor Parameters

| Parameter           | Type    | Description                                              | Default Value |
|---------------------|---------|----------------------------------------------------------|---------------|
| `dim`               | int     | The input dimension of the layer.                        | -             |
| `dropout`           | float   | Dropout probability.                                     | 0.1           |
| `eps`               | float   | Epsilon for numerical stability in LayerNorm.            | 1e-5          |
| `elementwise_affine`| bool    | Enables learning of affine parameters in LayerNorm.      | True          |

## Methods

### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of FusedDropoutLayerNorm.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor after applying dropout and layer normalization.
    """
```

## Examples

### Basic Usage

```python
import torch
from torch import nn

from zeta.nn import FusedDropoutLayerNorm

# Initialize the module
model = FusedDropoutLayerNorm(dim=512)

# Create a sample input tensor
x = torch.randn(1, 512)

# Forward pass
output = model(x)

# Check output shape
print(output.shape)  # Expected: torch.Size([1, 512])
```

### Integration in a Neural Network

```python
import torch
import torch.nn as nn

from zeta.nn import FusedDropoutLayerNorm


class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 512)
        self.fused_dropout_layernorm = FusedDropoutLayerNorm(512)

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_dropout_layernorm(x)
        return x


# Example
model = SampleModel()
input_tensor = torch.randn(10, 512)
output = model(input_tensor)
print(output.shape)  # Expected: torch.Size([10, 512])
```

### Custom Configuration

```python
import torch

from zeta.nn import FusedDropoutLayerNorm

# Custom configuration
dropout_rate = 0.2
epsilon = 1e-6
elementwise_affine = False

# Initialize the module with custom configuration
model = FusedDropoutLayerNorm(
    512, dropout=dropout_rate, eps=epsilon, elementwise_affine=elementwise_affine
)

# Sample input
x = torch.randn(1, 512)

# Forward pass
output = model(x)
print(output.shape)  # Expected: torch.Size([1, 512])
```

## Architecture and Working

The `FusedDropoutLayerNorm` module is architecturally a combination of two PyTorch layers: `nn.Dropout` and `nn.LayerNorm`. The fusion of these layers into a single module ensures that the operations are performed sequentially and efficiently, thereby reducing the computational overhead.

- **Dropout**: This operation randomly zeroes some of the elements of the input tensor with probability `dropout` during training. It helps prevent overfitting.
- **Layer Normalization**: This operation normalizes the input across the features. It stabilizes the learning process and accelerates the training of deep neural networks.

By integrating these two operations, `FusedDropoutLayerNorm` ensures a streamlined process where the dropout is applied first, followed by layer normalization. This design choice is made for computational efficiency and is particularly beneficial in transformer models and other deep learning architectures where both operations are frequently used.

## Purpose and Importance

The primary purpose of `FusedDropoutLayerNorm` is to provide a more efficient way to apply both dropout and layer normalization in a model. This efficiency is particularly crucial in

 large-scale models where computational resources and runtime are significant concerns. The module is designed to be versatile and can be easily integrated into various neural network architectures, especially those involving transformer models.

## Conclusion

The `FusedDropoutLayerNorm` module in PyTorch is a practical and efficient solution for models that require both dropout and layer normalization. Its fused architecture not only enhances computational efficiency but also simplifies the model design process. The module is flexible, allowing for easy customization and integration into diverse neural network architectures.

