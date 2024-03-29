# Module/Function Name: GatedResidualBlock

`class GatedResidualBlock(nn.Module):`

## Overview

The `GatedResidualBlock` is a subclass of the `nn.Module` which belongs to the PyTorch library. The main objective of this module is to implement a special variant of Residual Block structure which is commonly used in designing deep learning architectures.

Traditionally, a Residual Block allows the model to learn an identity function which helps in overcoming the problem of vanishing gradients in very deep networks. The `GatedResidualBlock` takes this a step further by introducing gating mechanisms, allowing the model to control the information flow across the network. The gate values, generated by the `gate_module`, determines the degree to which the input data flow should be altered by the first sub-block `sb1`.

This architecture promotes stability during the training of deep networks and increases the adaptability of the model to complex patterns in the data.

## Class Definition

The class definition for `GatedResidualBlock` is as follows:

```
class GatedResidualBlock(nn.Module):
    def __init__(self, sb1, gate_module):
        super().__init__()
        self.sb1 = sb1
        self.gate_module = gate_module
```

### Arguments

| Argument                           | Type         | Description                                                                                                      |
| ---------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------- |
| `sb1`                              | `nn.Module`  | The first sub-block of the Gated Residual Block.                                                                 |
| `gate_module`                      | `nn.Module`  | The gate module that determines the degree to which the input should be altered by the first sub-block `sb1`.    |

## Example: Usage of GatedResidualBlock

A simple usage of `GatedResidualBlock` is demonstrated below.

```python
import torch
import torch.nn as nn

from zeta.nn import GatedResidualBlock

# Define the sub-blocks
sb1 = nn.Linear(16, 16)
gate_module = nn.Linear(16, 16)

# Create the GatedResidualBlock
grb = GatedResidualBlock(sb1, gate_module)

# Sample input
x = torch.rand(1, 16)

# Forward pass
y = grb(x)
```

In the above example, both subblocks are simple linear layers. The input `x` is passed through the `GatedResidualBlock`, where it's processed by the `gate_module` and `sb1` as described in the class documentation.

## Method Definition

The method definition for `GatedResidualBlock` class is as follows:

```python
def forward(self, x: torch.Tensor):
    gate = torch.sigmoid(self.gate_module(x))
    return x + gate * self.sb1(x)
```

This method applies a standard forward pass to the input tensor `x` through the Gated Residual Block.

### Arguments

| Argument   | Type           | Description       |
| ---------- | -------------- | ----------------- |
| `x`        | `torch.Tensor` | The input tensor. |

### Returns

It returns a `torch.Tensor`, the output tensor of the gated residual block.

## Note

This module requires the inputs `sb1` and `gate_module` to be of `nn.Module` type. Any model architecture that extends `nn.Module` can be used as the sub-blocks. The gating mechanism helps to improve the model performance especially on complex and large data sets. 

If you encounter any issues while using this module, please refer to the official PyTorch documentation or raise an issue on the relevant GitHub issue page.
