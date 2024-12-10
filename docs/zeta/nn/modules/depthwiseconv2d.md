# Module/Function Name: DepthWiseConv2d

The `DepthWiseConv2d` class is a base class for all neural network modules. It serves as a fundamental element for creating deep learning models and contains multiple attributes that can be used for different applications and use cases. The `DepthWiseConv2d` class allows you to create deep neural networks by subclassing and utilizing its inbuilt features and capabilities. Additionally, it supports the nesting of modules and seamlessly incorporates submodules in a tree-like structure, providing flexibility and extensibility to the neural network architecture.

Example Usage:

```python
import torch.nn as nn
import torch.nn.functional as F

from zeta.nn import DepthWiseConv2d


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DepthWiseConv2d(1, 20, 5, padding=2, stride=1)
        self.conv2 = DepthWiseConv2d(20, 40, 5, padding=2, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth:`to`, etc.

Regarding the assignment of submodules in this class, the `__init__()` call to the parent class must be made prior to assigning child submodules.

Attributes:
- training: A boolean that represents whether this module is in training or evaluation mode.
    - Type: bool

Source Code:
```python
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                padding=padding,
                groups=dim_in,
                stride=stride,
                bias=bias,
            ),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)
```

In the above example, the DepthWiseConv2d class is defined with specified parameters `dim_in`, `dim_out`, `kernel_size`, `padding`, `stride`, and `bias`, where `dim_in` is the input dimension, `dim_out` is the output dimension, `kernel_size` is the size of the convolutional kernel, `padding` is the padding size, `stride` is the stride value, and `bias` is a boolean parameter to include bias in the convolution operation. The forward method applies this defined convolution operation to input `x` using `self.net` and returns the result.

By using the DepthWiseConv2d class with its specified parameters, you can create a deep neural network module that supports convolution operations with customizable input and output dimensions and kernel characteristics. With its comprehensive structure and modularity, DepthWiseConv2d facilitates the creation of sophisticated deep learning models.

For using this class in a more practical scenario, please refer to the usage example presented above and customize the class attributes to meet the requirements of your specific application or use case.
