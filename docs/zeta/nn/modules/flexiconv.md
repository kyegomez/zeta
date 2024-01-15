# Module/Function Name: FlexiConv

`class FlexiConv(nn.Module)`

FlexiConv is an experimental and flexible convolutional layer that adapts to the input data.

## Args

| Argument        | Description                                  | Data Type | Default Value |
|-----------------|----------------------------------------------|-----------|----------------|
| in_channels     | Number of channels in the input image        | int       | -              |
| out_channels    | Number of channels produced by the convolution | int     | -              |
| kernel_size     | Size of the convolving kernel                | int/tuple | -              |
| stride          | Stride of the convolution                    | int/tuple | 1              |
| padding         | Zero-padding added to the input              | int/tuple | 0              |
## Example

```python
import torch 
from zeta.nn import FlexiConv

flexi_conv = FlexiConv(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
input_tensor = torch.randn(1, 3, 224, 224)  # Example input batch
output = flexi_conv(input_tensor)
output.shape
```

## Purpose

FlexiConv is aimed at providing a flexible convolutional layer that adapts to the input data using parameterized Gaussian functions to weigh the importance of each pixel in the receptive field and applies a depthwise separable convolution for efficiency.

## Functionality
The FlexiConv class encapsulates a flexible convolutional layer that uses Gaussian functions to weigh the importance of each pixel in the receptive field. It applies a depthwise separable convolution to efficiently process input data. The user can specify the number of input and output channels, kernel size, and stride, among other parameters.

## Usage
The `FlexiConv` layer can be instantiated by passing the required arguments and then used to process input tensors.

Example 1:
```python
import torch 
from zeta.nn import FlexiConv

flexi_conv = FlexiConv(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
input_tensor = torch.randn(1, 3, 224, 224)
output = flexi_conv(input_tensor)
output.shape
```

Example 2:
```python
import torch 
from zeta.nn import FlexiConv


flexi_conv = FlexiConv(in_channels=3, out_channels=64, kernel_size=3, stride=(2,2), padding=1)
input_tensor = torch.randn(1, 3, 224, 224)
output = flexi_conv(input_tensor)
output.shape
```

Example 3:
```python
import torch 
from zeta.nn import FlexiConv


flexi_conv = FlexiConv(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,2), padding=1)
input_tensor = torch.randn(1, 3, 224, 224)
output = flexi_conv(input_tensor)
output.shape
```
## References
Provide any references to further information or research papers related to the FlexiConv module or framework.

## Additional Information
Provide any tips or additional details that may be useful for using the FlexiConv module effectively.

By documenting the FlexiConv example, the document provides an in-depth explanation of its purpose, usage, functionality, and examples to ensure the user understands how to effectively leverage the FlexiConv module.
