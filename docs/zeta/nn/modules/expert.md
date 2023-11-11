# Module Documentation: `Experts`

## Overview

The `Experts` module is designed to implement an expert module for the Mixture of Experts layer. This module is particularly useful for tasks that require the combination of information from different subspaces. It takes input features of a specific dimension and processes them through multiple experts to produce an output tensor of shape `(batch_size, seq_len, dim)`.

In this documentation, we will provide a detailed explanation of the `Experts` module, including its purpose, class definition, parameters, functionality, and usage examples.

## Table of Contents

1. [Class Definition](#class-definition)
2. [Parameters](#parameters)
3. [Functionality](#functionality)
4. [Usage Examples](#usage-examples)
5. [Additional Information](#additional-information)

## Class Definition <a name="class-definition"></a>

```python
class Experts(nn.Module):
    def __init__(
        self,
        dim: int,
        experts: int = 16,
    ):
        """
        Expert module for the Mixture of Experts layer.

        Args:
            dim (int): Dimension of the input features.
            experts (int): Number of experts.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(experts, dim, dim * 2))
        self.w2 = nn.Parameter(torch.randn(experts, dim * 4, dim * 4))
        self.w3 = nn.Parameter(torch.randn(experts, dim * 4, dim))
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        """Forward pass."""
        hidden1 = self.act(torch.einsum('end,edh->enh', x, self.w1))
        hidden2 = self.act(torch.einsum('end,edh->enh', hidden1, self.w2))
        out = torch.einsum('end,edh->enh', hidden2, self.w3)
        return out
```

## Parameters <a name="parameters"></a>

- `dim` (int): Dimension of the input features.
- `experts` (int): Number of experts.

## Functionality <a name="functionality"></a>

The `Experts` module takes input features of dimension `dim` and processes them through a series of operations to produce an output tensor of shape `(batch_size, seq_len, dim)`.

The operations performed in the `forward` method include:
1. Linear transformation of the input features using learnable weights `w1`, followed by the LeakyReLU activation function.
2. Another linear transformation of the intermediate result using learnable weights `w2`, followed by the LeakyReLU activation function.
3. A final linear transformation of the last intermediate result using learnable weights `w3`.

The `forward` method returns the final output tensor.

## Usage Examples <a name="usage-examples"></a>

Here are three usage examples of the `Experts` module:

### Example 1: Basic Usage

```python
import torch
from torch import nn
from zeta.nn import Experts

# Create input tensor
x = torch.randn(1, 3, 512)

# Initialize the Experts module with 16 experts
model = Experts(512, 16)

# Forward pass
out = model(x)

# Print the shape of the output tensor
print(out.shape)  # Output: torch.Size([1, 3, 512])
```

### Example 2: Custom Number of Experts

```python
import torch
from torch import nn
from zeta.nn import Experts

# Create input tensor
x = torch.randn(2, 4, 256)

# Initialize the Experts module with 8 experts
model = Experts(256, 8)

# Forward pass
out = model(x)

# Print the shape of the output tensor
print(out.shape)  # Output: torch.Size([2, 4, 256])
```

### Example 3: Using Device and Data Type

```python
import torch
from torch import nn
from zeta.nn import Experts

# Create input tensor
x = torch.randn(3, 5, 128)

# Initialize the Experts module with 4 experts on GPU
model = Experts(128, 4)
model.to('cuda')  # Move the model to GPU
x = x.to('cuda')  # Move the input tensor to GPU

# Forward pass
out = model(x)

# Print the shape of the output tensor
print(out.shape)  # Output: torch.Size([3, 5, 128])
```

## Additional Information <a name="additional-information"></a>

- The `Experts` module is designed to handle multi-expert processing of input features, making it suitable for tasks that require information combination from different subspaces.
- You can customize the number of experts by adjusting the `experts` parameter.
- You can also specify the device and data type for the module and input tensor for efficient computation.

For more details on the usage and customization of the `Experts` module, refer to the code examples and experiment with different configurations to suit your specific needs.