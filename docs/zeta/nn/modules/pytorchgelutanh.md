# PytorchGELUTanh

## Overview

The `PytorchGELUTanh` class in Python is a fast C implementation of the tanh approximation of the GeLU activation function. This implementation is meant to be faster and as effective as other implementations of GeLU (Gaussian Error Linear Units) function like NewGELU and FastGELU. However, it is not an exact numerical match to them due to possible rounding errors.

This documentation provides an in-depth guide to using the `PytorchGELUTanh` class. It includes general information about the class, the method documentation, and various usage examples.

## Introduction

In Neural Networks, activation functions decide whether a neuron should be activated or not by calculating the weighted sum and adding bias with it. One of these activation functions is the Gaussian Error Linear Units (GeLU) function. GeLU function approximates the cumulative distribution function of the standard Gaussian distribution and helps in faster learning during the initial phase of training.

The `PytorchGELUTanh` class provides a fast C implementation of the tanh approximation of the GeLU activation function.

## Class Definition

```python
class PytorchGELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.12.0"):
            raise ImportError(
                f"You are using torch=={torch.__version__}, but torch>=1.12.0"
                " is required to use PytorchGELUTanh. Please upgrade torch."
            )

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.gelu(input, approximate="tanh")
```

## General Information

The `PytorchGELUTanh` class only requires PyTorch version 1.12.0 or higher. 

This class contains the following methods:

| Method  | Definition |
| --- | --- |
| `__init__` | This is the constructor method for the `PytorchGELUTanh` class in which the superclass is initialized and a check is made to ensure that the version of PyTorch being used supports the class. If not, an import error is raised. |
| `forward` | This method applies the tanh approximation of the GeLU active function to the provided tensor input. |

The `forward` method takes in a tensor as an input argument and returns a tensor as an output. The input and output tensors are of the same size.

## Usage Examples

### Example 1: Basic Usage

In this basic example, we create an instance of the `PytorchGELUTanh` class and pass a tensor to its `forward` method to apply the tanh approximation of the GeLU function.

```python
# Import necessary libraries
import torch
from packaging import version
from torch import Tensor, nn
from torch.nn.functional import gelu

from zeta.nn import PytorchGELUTanh

# Create an instance of the PytorchGELUTanh class.
gelutanh = PytorchGELUTanh()

# Create a tensor.
x = torch.randn(3)

# Print the tensor before and after applying the GeLU Tanh activation function.
print("Before: ", x)
print("After: ", gelutanh.forward(x))
```

### Example 2: Application to Deep Learning

The `PytorchGELUTanh` class can be used in place of traditional activation functions in deep learning models. Here is an example of its usage in a feed-forward neural network.

```python
# Import necessary libraries
import torch
from torch import Tensor, nn
from torch.nn.functional import gelu

from zeta.nn import PytorchGELUTanh


# Define a feed-forward neural network with 2 layers and the PytorchGELUTanh activation function
class FeedForwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)  # 10 input neurons, 20 output neurons
        self.gelu = PytorchGELUTanh()  # Our custom activation function
        self.fc2 = nn.Linear(20, 1)  # Final layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)  # Apply the PytorchGELUTanh activation
        x = self.fc2(x)
        return x


# Instantiate the model
model = FeedForwardNN()

# Print the model architecture
print(model)
```

This completes the documentation for the `PytorchGELUTanh` Python class, but feel free to reference the official [PyTorch documentation](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.gelu) and ensure you are using a version of PyTorch that is compatible with this class.
