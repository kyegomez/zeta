# AccurateGELUActivation

## Overview
The AccurateGELUActivation class is a part of the PyTorch library's nn.Module. This class allows us to apply the Gaussian Error Linear Unit (GELU) approximation that is faster than the default and more accurate than QuickGELU. This can be useful in situations where the default GELU is considered computationally expensive or its speed could be an issue. The implementation of this class comes as a support for MEGA, which stands for Moving Average Equipped Gated Attention, in neural networks.

The class has been designed following the work on GELUs available at: [https://github.com/hendrycks/GELUs](https://github.com/hendrycks/GELUs)

## Class Definition
Here is a look at the parameters and methods used in the `AccurateGELUActivation` class:

```python
class AccurateGELUActivation(nn.Module):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs
    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (
                1
                + torch.tanh(
                    self.precomputed_constant
                    * (input + 0.044715 * torch.pow(input, 3))
                )
            )
        )
```

The class does not require any parameters during initialization. Here are the explanations for the various attributes and methods in the class:

| Method/Attribute | Description | Argument |
| --- | --- | --- |
| `__init__` | This is the constructor method that gets called when an object is created from the class. | None |
| `forward` | This method is a PyTorch standard for forward propagation in a Module or a neural network layer. It accepts a tensor input and returns a tensor. | `input: Tensor` |

## Class Usage
Now, let's look at some examples of how to use this class.

### Example 1: Basic Usage
```python
import torch
from torch.nn import Module
import math
from torch import Tensor
from zeta import AccurateGELUActivation
        
# Create an instance of the class
gelu_activation = AccurateGELUActivation()

# Create a PyTorch tensor
input = torch.tensor([[-1.0, -0.1, 0.1, 1.0], [0.5, -0.2, -2.1, 3.2]], dtype=torch.float32)

# Use the AccurateGELUActivation instance to activate the input
output = gelu_activation(input)

print(output)
```
This example demonstrates the functionalities of the AccurateGELUActivation module for a defined two-dimensional input tensor.

### Example 2: Applying on Neural Network
The AccurateGELUActivation module can also be used as an activation layer in a PyTorch model.

```python
import torch
from torch.nn import Module, Linear
import math
from torch import Tensor
from zeta.nn import AccurateGELUActivation

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(10, 5)
        self.fc2 = Linear(5, 2)
        self.activation = AccurateGELUActivation()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x     

# Create a model from the neural network class
model = Net()

input = torch.randn(3, 10)

# Pass the input to the model
output = model(input)

print(output)
```
This example shows how the AccurateGELUActivation module can be integrated as a layer in a neural network model to perform activation on the intermediate outputs of the neural network model.

**Note:** Please remember, understanding what activation functions like GELU can do, what benefits they can bring to your architecture, is crucial before applying it to your models.
