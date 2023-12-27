# HighwayLayer

## Module Introduction

`HighwayLayer` is a class implemented in PyTorch that provides an easy way to include Highway layers in your model. The Highway layer is a type of artificial neural network (ANN) that aids in remembering or carrying information across several layers. It consists of a normal layer and a gate layer.

It addressed the vanishing gradient problem typically found in the training of deep networks. With the application of a gating mechanism, the Highway layer dynamically routes signals through paths for different samples and different layers without harming the optimization process.

This document provides details on how to use this class, its methods, properties, and examples for better understandings.

## Class Definition

```python
class HighwayLayer(nn.Module):
```

Inherits from the `nn.Module` class which is the base class for all neural network modules in PyTorch.

## Parameters

- `dim` (int): The dimension of the input tensor to the layer and the output of the layer.

## Methods

### `__init__(self, dim)`

Initializes a `HighwayLayer` instance with a specified `dim`.

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| dim       | int  | The input and output dimension of the layer |

### `forward(self, x)`

Performs a forward pass through the `HighwayLayer`.

Parameters:

| Parameter | Type           | Description       |
|-----------|----------------|-------------------|
| x         | torch.Tensor   | The input tensor  |

Returns:

`torch.Tensor`: The output tensor.

## Source Code

```python
import torch.nn as nn
import torch.nn.functional as F

class HighwayLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.normal_layer = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normal_result = F.relu(self.normal_layer(x))
        gate = torch.sigmoid(self.gate(x))
        return gate * normal_result + (1 - gate) * x
```

## Usage Examples

### Example 1: Simple model with single HighwayLayer

```python
import torch
from zeta.nn import HighwayLayer

# Initialize HighwayLayer with dimension 50
layer = HighwayLayer(50)

# Random input tensor of shape (10, 50)
input_tensor = torch.randn(10, 50)
output_tensor = layer(input_tensor)

print(output_tensor.shape)  # Expected shape (10, 50)
```

### Example 2: Model with Multiple Highway Layers

```python
import torch
from zeta.nn import HighwayLayer

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = HighwayLayer(50)
        self.layer2 = HighwayLayer(50)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Initialize model and input tensor
model = MyModel()
input_tensor = torch.randn(10, 50)

# Forward pass
output_tensor = model(input_tensor)

print(output_tensor.shape)  # Expected output: torch.Size([10, 50])
```

### Example 3: Model with HighwayLayer and Other Types of Layers

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = HighwayLayer(50)
        self.layer2 = nn.Linear(50, 20)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Initialize model and input tensor
model = MyModel()
input_tensor = torch.randn(10, 50)

# Forward pass
output_tensor = model(input_tensor)

print(output_tensor.shape)  # Expected output: torch.Size([10, 20])
```

Application of HighwayLayer can greatly enhance the learning of deep neural networks by allowing the direct forward flow of information unimpeded thereby solving the vanishing gradient problem.
