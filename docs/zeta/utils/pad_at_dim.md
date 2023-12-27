# pad_at_dim

# Zeta Utils Library Documentation

## Module Function: pad_at_dim
***pad_at_dim*** is a utility function in the Zeta Utilities Library for padding tensors at a specified dimension to match the desired dimensions. This function builds on Pytorch's built-in function ***F.pad()*** providing additional configurability to specify the dimension at which padding is done. The provided padding is appended at the end of the input tensor's specified dimension.

## Function Signature
```python
def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)
```

## Important Parameters Definition
| Parameters   | Type   | Description                                                                                                        |
| :----------- | :----- | :----------------------------------------------------------------------------------------------------------------- |
| t            | Tensor | Input tensor in the PyTorch format.                                                                                |
| pad          | Tuple  | Padding size for each side of the tensor's dimension. Padding format is (pad_left, pad_right).                    |
| dim          | Integer| The dimension at which padding is performed. By default, it's -1, which indicates the last dimension.  |
| value        | Float  | The padding value. Default is 0.0.                                                                                 |

## Functionality and Usage

The ***pad_at_dim*** function performs padding operation on PyTorch tensors at the specified dimension using Pytorch's built-in ***F.pad*** function. It takes into account both positive and negative dimension indices. While positive indices perform the padding from the first dimension, negative indices do the padding starting from the last dimension.

Creating the zeros needed to fill the rest of the parameters of the PyTorch's F.pad function, the function internally calculates how many zeros are needed, given the dimension. 

Subsequently, it calls F.pad function using the calculated zeros, the desired padding and value to add padding in the given tensor at the specified dimension.

## Function Examples

Let's dive in into few examples to understand how the module can be used.

### Example 1: Padding the last dimension

```python
import torch
from torch.nn import functional as F
from zeta.utils import pad_at_dim

# Create a tensor
t = torch.tensor([[7, 8, 
