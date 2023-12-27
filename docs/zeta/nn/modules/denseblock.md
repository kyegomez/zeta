# Class Name: DenseBlock

The `DenseBlock` class is a type of PyTorch `nn.Module`. This allows for complicated neural network architectures to be defined with individual abstracted layers. The class gets its name from the dense connections made in the forward propagation, which involve concatenating the output of the `submodule` with the original input. 

For the following documentation, the DenseBlock class is used as an example of such constructions. 

While this class might seem simple, understanding how it works is fundamental to define, compile, and use your own custom PyTorch models. 

It has two main methods, the `__init__()` method and the `forward()` method.

### Method: \_\_init__(self, submodule, *args, **kwargs)

The `__init__()` method is the initializer method of the DenseBlock class. It is called when an object (an instance of the class) is created. 

This method sets an attribute of the DenseBlock object to be the `submodule` input, which is assumed to be some `nn.Module` instance.

The method signature is:

    def __init__(self, submodule, *args, **kwargs)

#### Arguments

|Name|Type|Description|
|---|---|---|
|submodule|nn.Module|The module that will be applied in the forward pass.|
|args|Variable length argument list|Unused in this implementation, but allows for extra position arguments.|
|kwargs|Arbitrary keyword arguments|Unused in this implementation, but allows for extra keyword arguments.|

The `submodule` argument should be an initialized instance of the `nn.Module` subclass you want to apply. 

The `args` and `kwargs` arguments are not currently used in DenseBlock. 

### Method: forward(self, x: torch.Tensor) -> torch.Tensor

The `forward()` method is called during the forward propagation of the neural network. 

It applies the module operation to the input tensor `x` and concatenates the input tensor `x` with the output of the `submodule`.

The method signature is:

    def forward(self, x: torch.Tensor) -> torch.Tensor

#### Arguments

|Name|Type|Description|
|---|---|---|
|x|torch.Tensor|The input tensor to the module.|

Returns a tensor, which is the input tensor concatenated with the processed input tensor via the `submodule`.

## Usage Examples

Here are some examples showing how to use the DenseBlock class. These examples will include the necessary imports, data creation, and model instantiation following PyTorch conventions:

### Example 1: Basic Usage with a Linear Layer

In this example, the `DenseBlock` will include a Linear layer as submodule.

```python
import torch
import torch.nn as nn
from torch.autograd import Variable
from zeta.nn import DenseBlock

# Defining submodule
lin_layer = nn.Linear(5, 10)

# Defining DenseBlock
dense_block = DenseBlock(lin_layer)

# Creating a random tensor of shape [10, 5]
random_tensor = Variable(torch.randn(10, 5))

# Applying DenseBlock
output = dense_block(random_tensor)
```

In this example, an input tensor of shape [10,5] is given to a dense block with a linear layer. The input will have shape [10,5] and the output of the linear layer will have shape [10,10], resulting in the output of the dense block to have shape [10,15].

### Example 2: Using DenseBlock in a Multilayer Neural Network

In this example, a 2-layer neural network using Dense Blocks is shown. The first layer is a Dense Block with a Linear module transforming with dimensions (10 to 5), and the second layer is a standard Linear layer transforming the output dimensions (15 to 1).
```python
import torch.nn.functional as F

# Defining a custom model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = DenseBlock(nn.Linear(10, 5))
        self.layer2 = nn.Linear(15, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Initializing the model
net = Net()

# Creating a random tensor of shape [32, 10]
data = Variable(torch.randn(32, 10))

# Forward propagation
output = net(data)
```

In this second example, a data batch with `32` samples and input dimensionality of `10` is given to a `Net` neural network with dense connections in their first layer. The final output shape is [32, 1]. 

### Example 3: DenseBlock with Convolutional Layer

Lastly, this example shows how to use DenseBlock inside a Convolutional Neural Network:
```python
import torch
import torch.nn as nn
from zeta.nn import DenseBlock

cnn = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    DenseBlock(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(128, 10),
)

x = torch.randn(1, 1, 224, 224)
output = cnn(x)
```

Here, a 2D convolutional layer is used as the submodule within the DenseBlock. The DenseBlock receives a tensor with shape [64, 224, 224] as input, applies the convolutional layer (keeping the same shape), and then concatenates the input and the output along the channel dimension, resulting in a tensor with shape [128, 224, 224].
