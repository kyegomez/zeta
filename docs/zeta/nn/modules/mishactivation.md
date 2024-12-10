# MishActivation

This is the official documentation for the Mish Activation class implementation in PyTorch. 
This document will cover the details of implementing Mish Activation function and the ways to use it.

## Mish Activation Function: Introduction

Mish Activation is a novel approach to optimizing and enhancing the performance of neural network models by using a new self-regularized, non-monotonic activation function known as "Mish". Mish aims to promote better gradient flow for deep networks, while also distinguishing extreme gradient values for generalization in deep networks.

For a more deep understanding of the function you can refer to the official paper by Diganta Misra that presents and discusses the Mish activation function, ["Mish: A Self Regularized Non-Monotonic Neural Activation Function"](https://arxiv.org/abs/1908.08681).

There is also a GitHub repo available for detailed information and research related to Mish Activation function [Here](https://github.com/digantamisra98/Mish).

## Class Definition

```python
class MishActivation(nn.Module):
    """
    A pytorch implementation of mish activation function.
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9.0"):
            self.act = self._mish_python
        else:
            self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)
```

## Class Arguments & Methods

### Arguments
Mish Activation function does not take any explicit argument other than the input tensor. 

### Methods

#### `__init__(self)`

This is the initialization method where mish activation function checks for PyTorch version and based on the version, decides whether to use PyTorch built-in Mish Activation function or fall back to its own python implementation of Mish Activation function.

#### `_mish_python(self, input: Tensor) -> Tensor`

The fallback python implementation of Mish Activation function that multiplies the input with a hyperbolic tanh of a softplus function of input.

- Parameters:
  - `input: Tensor`: The tensor on which the activation function will be applied.

- Returns:
    - `Tensor`: The modified tensor after applying the activation function.

#### `forward(self, input: Tensor) -> Tensor`

The forward method applies mish activation on the input tensor

- Parameters:
  - `input: Tensor`: The tensor on which the activation function will be applied.

- Returns:
    - `Tensor`: The modified tensor after applying the activation function.

## Usage Examples

This module requires PyTorch and Python 3.6 or above.
### Example 1: Importing the module and Applying the Mish Activation function

```python
from packaging import version
from torch import Tensor, nn
from torch.nn import functional as F

from zeta.nn import MishActivation

input_tensor = Tensor([[-0.6, 0.7], [1.2, -0.7]])
mish = MishActivation()
print(mish.forward(input_tensor))
```
### Example 2: Using Mish Activation for Neural Network Layers

The Mish Activation function can also be applied in Neural Network layers using PyTorch.

```python
import torch
from packaging import version
from torch import Tensor, nn
from torch.nn import functional as F

from zeta.nn import MishActivation


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Sequential(
            nn.Linear(26, 256), MishActivation(), nn.Linear(256, 10), MishActivation()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layer(x)
        return logits


model = NeuralNetwork()
# Following lines shows how to use the model, given the input tensor, `X`.
# output = model(X)
```
## References

- [Packaging](https://pypi.org/project/packaging/)
- [PyTorch](https://pytorch.org/docs/stable/torch.html)
- [Arxiv Article for Mish Activation](https://arxiv.org/abs/1908.08681)
- [GitHub repo for MishActivation](https://github.com/digantamisra98/Mish)
