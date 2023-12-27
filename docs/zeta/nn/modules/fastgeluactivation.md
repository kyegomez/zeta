# FastGELUActivation

This is a comprehensive documentation for `FastGELUActivation`, a class of the SWARMS library. 

## Overview 
FastGELUActivation is a class implemented in the SWARMS library that introduces an optimized approach to computing Gaussian Error Linear Units (GELUs). It's based on a faster approximation of the GELU activation function, which is generally more accurate than QuickGELU. 

GELU activation is frequently used in many machine learning applications, particularly deep learning models, to add non-linearity to the operations. Such activation functions help models represent a wider range of phenomena and thus yield more robust and accurate results. For reference on GELUs, please refer to [Hendrycks GELUs](https://github.com/hendrycks/GELUs).

## Class Definition and Functionality 
FastGELUActivation is a class in PyTorch's nn.Module that overrides the forward method to provide a new functionality. Below is the class definition of `FastGELUActivation`.

```python
class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate.
    """
    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    input * 0.7978845608 * (1.0 + 0.044715 * input * input)
                )
            )
        )
```       

## Parameters
The `FastGELUActivation` class uses only one parameter as input in its forward method.

| Parameter | Type | Description |
| - | - | - |
| `input` | Tensor | The input tensor that the forward pass needs to compute over.|

### Inputs
The input that `FastGELUActivation` takes is a PyTorch Tensor, which holds the values that the activation function computes.

### Outputs
The forward method of `FastGELUActivation` returns a new tensor, which is the result of applying the FastGELU activation operation to the input tensor.

## Usage and Workflow 
Using `FastGELUActivation` involves creating an instance of the class and then using that instance to call the class's `forward` method with an appropriate input Tensor.

### Example Usage
In this example, we'll create a simple tensor and apply the `FastGELUActivation` activation function to it.

```python
import torch
from torch import nn, Tensor
from zeta import FastGELUActivation

# Create an instance of FastGELUActivation
activation = FastGELUActivation()

# Create a tensor
tensor = torch.randn((5,5), dtype=torch.float32)

# Apply FastGELUActivation
result = activation.forward(tensor)

print(result)
```
### Working with Real World Data Example
Assuming we're building a neural network that uses the `FastGELUActivation` as its activation function in one of the layers:

```python
import torch.nn as nn
from zeta import FastGELUActivation

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(in_features=784, out_features=512)
        self.layer2 = nn.Linear(in_features=512, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=10)
        self.activation = FastGELUActivation()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x

model = NeuralNet()
```

In this example, we have a simple feedforward neural network with two layers, and it uses `FastGELUActivation` for the intermediate layers.

## Additional information & Tips
The `FastGELUActivation` is a faster approximation of the GELU activation operation, but not always the most accurate. Depending on your use case and performance requirements, you may want to use a more robust but slower activation function.

Make sure to have a profound understanding of the dataset and context before deciding on the activation function.
