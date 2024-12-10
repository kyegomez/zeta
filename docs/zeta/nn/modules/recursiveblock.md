# RecursiveBlock


Zeta is a python library that makes use of Pytorch for implementing several classes and functions related to swarm optimization tasks. This documentation will be focusing on the `RecursiveBlock` class in the `swarm` Pytorch-based library. This class's main functionality is to recursively apply a given module a specified number of times to an input tensor.

The RecursiveBlock is, therefore, a versatile class that allows for a wide range of operations to be performed on your data by reiterating the application of an operation or set of operations encapsulated in a module.

## Class Definition
Here is the code structure of the RecursiveBlock class:

```python
import torch
from torch import nn


class RecursiveBlock(nn.Module):
    def __init__(self, modules, iters, *args, **kwargs):
        super().__init__()
        self.modules = modules
        self.iters = iters

    def forward(self, x: torch.Tensor):
        for _ in range(self.iters):
            x = self.modules(x)
        return x
```

## Parameters and Arguments
Let's discuss the function definitions, parameters, and return types of `RecursiveBlock's` methods.

### `__init__` Constructor Method: 
This method initializes the `RecursiveBlock` object.
Parameters of this constructor are:

| Parameter | Type | Description |
|-----------|------|-------------|
| `modules` | torch.nn.Module | The module to be applied recursively. |
| `iters`   | int             | The number of iterations to apply the module. |
| `*args`   | list            | Variable length argument list. |
| `**kwargs`| dict            | Arbitrary keyword arguments. |

### `forward` Method:
This method is responsible for the forward pass of the block.
Parameters of this method are:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x`       | torch.Tensor | The input tensor.|

Return Type: **torch.Tensor** : The output tensor after applying the module recursively.

## Usage Examples

### Example 1:
Utilizing two convolutional layers from Pytorch's nn library recursively

```python
import torch
from torch import nn

from zeta import RecursiveBlock

conv_module = nn.Sequential(
    nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 20, 5), nn.ReLU()
)

block = RecursiveBlock(conv_module, iters=2)

x = torch.randn(1, 20, 10, 10)
output = block(x)
```

### Example 2:
Implementing the RecursiveBlock class with a simple, custom module

```python
class AddTen(nn.Module):
    def forward(self, x):
        return x + 10


block = RecursiveBlock(AddTen(), iters=3)
output = block(torch.tensor(1.0))  # output -> tensor(31.)
```

### Example 3:
Using RecursiveBlock with a Linear Layer and a sigmoid activation function

```python
import torch
from torch import nn

from zeta import RecursiveBlock

linear_module = nn.Sequential(
    nn.Linear(128, 64),
    nn.Sigmoid(),
)

block = RecursiveBlock(linear_module, iters=3)

x = torch.randn(16, 128)
output = block(x)
```

## Additional Information and Tips

1. The `modules` parameter in `RecursiveBlock` is not limited to built-in PyTorch modules. It can also be a custom PyTorch nn.Module defined by the user.

2. The `iters` parameter can be adjusted as per the requirement of the task. More iterations might lead to a deeper feature extraction and can sometimes lead to better performance, but can also increase the computation time.

Thus, RecursiveBlock is a simple yet powerful class providing the abstraction of repeated module application, making iterating through a module multiple times a straightforward task. It enables cleaner, more readable code for models involving repetition of a similar structure or block, ushering rich flexibility into the hands of the programmer.
