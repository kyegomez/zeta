# Module/Function Name: WSConv2d

## Overview and Introduction
WSConv2d is weight standardization Conv2d layer, that inherits from `nn.Conv2d` and adds weight standardization to the convolutional layer. It normalizes the weights of the convolutional layer to have zero mean and unit variance along the channel dimension. This helps in stabilizing the training process and improving generalization.

### Class: WSConv2d
#### Definition:
```python
class WSConv2d(nn.Conv2d):
```

##### Parameters:
Parameters | Description
--- | ---
in_channels (int) | Number of input channels
out_channels (int) | Number of output channels
kernel_size (int) | Size of the convolutional kernel
stride (float, optional) | Stride of the convolution. Default is 1
padding (int or tuple, optional) | Padding added to the input. Default is 0
dilation (int, optional) | Spacing between kernel elements. Default is 1
groups (int, optional) | Number of blocked connections from input channels to output channels. Default is 1
bias (bool, optional) | If True, adds a learnable bias to the output. Default is True
padding_mode (str, optional) | Type of padding. Default is "zeros"

## Method: init
```python
def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: float = 1,
    padding=0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
)
```
In the `__init__` method, the `WSConv2d` class initializes the convolutional layer with various attributes including in_channels, out_channels, kernel_size, stride, and bias. 

## Additional Properties:
- **gain**: nn.Parameter, shape (output_channels, 1, 1, 1), initialized to ones
- **eps**: register_buffer for a tensor with a single value of 1e-4
- **fan_in**: register_buffer for a tensor with the value equal to the number of weight parameters

## Method: standardized_weights
```python
def standardized_weights(self) -> Tensor
```
The `standardized_weights` method calculates the standardized weights using weight standardization, which makes use of mean and variance along each channel of the weights tensor.

## Method: forward
```python
def forward(self, x: Tensor) -> Tensor
```
The `forward` method convolves the input tensor `x` with standardized weights.

Example Usage:
```python
import torch

from zeta.nn import WSConv2d

# Instantiate a WSConv2d layer
ws_conv2d = WSConv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# Create a random input tensor
x = torch.randn(1, 3, 32, 32)

# Apply the WSConv2d layer
output = ws_conv2d(x)

print(output.shape)
```
Note: Modify the input and parameter values based on your use case and neural network architecture.

