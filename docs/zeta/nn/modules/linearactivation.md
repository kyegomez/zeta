# LinearActivation



The LinearActivation class belongs to the `nn.Module` in PyTorch which is a standard base class for all neural network modules. The class LinearActivation is a child class that inherits the functionalities of its parent class `nn.Module`. This class represents the linear activation function in the neural networks; sometimes also referred to as the identity function. The idea here is to return the input without applying any transformation, which means that the output of this function is the same as the input.

The source code is as follows:

```python
import torch.nn as nn
from torch import Tensor

from zeta.nn import LinearActivation


class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e., forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        return input
```

### Method details
**Method Name:** `forward`

This method executes the forward pass, in other words, it makes a forward pass from input to the output. The `forward` is an abstract method in superclass `nn.Module` and must be defined by each layer. 

**Arguments:**

| Argument Name | Type     | Description                                         |
|---------------|----------|-----------------------------------------------------|
| input         | Tensor   | Input tensor to which the linear activation is applied |

**Returns:**

`Tensor`: The output tensor identical to the input tensor. 

## Usage Example 1
```python
import torch
import torch.nn as nn
from torch import Tensor

from zeta.nn import LinearActivation

linear_activation = LinearActivation()

# random tensor of size 4
input_tensor = torch.randn(4)
print("Input tensor: ", input_tensor)

output_tensor = linear_activation(input_tensor)
print("Output tensor: ", output_tensor)
```
In this example, the `LinearActivation` class is instantiated first followed by generating a random tensor of size 4. This random tensor is passed to the instantiated `LinearActivation` class, and the result will be an identical tensor to the input, as expected.

## Usage Example 2

```python
import torch
import torch.nn as nn
from torch import Tensor

from zeta.nn import LinearActivation

# create an instance of the class LinearActivation
linear_activation = LinearActivation()

# define a tensor of ones
input_tensor = torch.ones(10)
print("Input tensor: ", input_tensor)

# pass the tensor of ones through the LinearActivation
output_tensor = linear_activation(input_tensor)
print("Output tensor: ", output_tensor)
```
In the second example, we create an input tensor of ones of size 10. When this tensor is passed through the `LinearActivation`, we expect an identical tensor of ones for the output. We print the output tensor to verify this.

## Usage Example 3

```python
import torch
import torch.nn as nn
from torch import Tensor

from zeta.nn import LinearActivation

linear_activation = LinearActivation()

# create a tensor with numbers from 1 to 10
input_tensor = torch.arange(1, 11).float()
print("Input tensor: ", input_tensor)

output_tensor = linear_activation(input_tensor)
print("Output tensor: ", output_tensor)
```
In the third example, we create an input tensor with numbers from 1 to 10. We then pass this tensor through the `LinearActivation`. Because the `LinearActivation` doesn't actually perform any mathematical transformations, the expected output tensor will be identical to the input tensor.
