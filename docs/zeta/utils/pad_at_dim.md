# pad_at_dim

# Module Name: pad_at_dim

## Introduction

The `pad_at_dim` function is a utility function used to apply padding to a tensor at a specified dimension. Padding is added to the edges of an input tensor and it's commonly used in convolutional neural networks where the input is often padded to control the output size of feature maps. This utility function is very useful to PyTorch users as it allows to add padding flexibly at any dimension, specified by the user.

The tensor padding is particularly useful in the context of image processing where it is often needed to apply the convolution kernel to bordering pixels of an input image. In the context of natural language processing tasks, padding is used when batching together sequences of different lengths, and can be used to ensure that all sequences in a batch are the same length.

## Function Definition

The function `pad_at_dim` has the following signature:

```python
def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)
```

## Parameters

| Parameter | Type      | Description | Default value |
| --------- | --------- | ----------- | ------------- |
| t         | torch.Tensor  | Input tensor to which padding will be applied. | NA |
| pad       | tuple | Number of values padded to the edges of each dimension, provided as a tuple in the format (padLeft, padRight) for each dimension. | NA |
| dim       | int     | Dimension at which padding will be added. Negative integer counts from the last dimension (-1 is the last dimension, -2 is the second last dimension, and so on). | -1 |
| value     | float   | Value for the padded elements. | 0.0 |

## Return

The function returns a tensor `t` padded at the specified `dim` with the given `value`. The padding size is specified by the `pad` parameter.

## Detailed Explanation & Usage

The `pad_at_dim` function uses the PyTorch `nn.functional.pad()` method to add padding to the tensor. It starts by determining the number of dimensions from the right of the tensor for which padding will be applied, stored in `dims_from_right`. It then creates the `zeros` tuple which has the number of zeros corresponding to the decided padding. Finally, the `pad` and `zeros` tuples are concatenated and used as input to the `nn.functional.pad()` method along with the original tensor and padding value.

Dimensions in PyTorch are 0-index based, therefore 0 refers to the first dimension and -1 refers to the last dimension. When the padding size (pad) is a tuple, the padding applied is symmetric for each dimension. If pad is an int, the same amount of padding is applied at both ends of the tensor.

The value parameter is used to fill in the new elements created due to padding operation.

### Usage Examples

Let's look at some examples demonstrating the `pad_at_dim` function:

1. Basic usage:

```python
import torch
from torch.nn import functional as F

# Define a tensor
t = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Call pad_at_dim
result = pad_at_dim(t, pad=(1, 1), dim=-1, value=0)

print(result)
```

Output:
```
tensor([[0, 1, 2, 3, 0],
        [0, 4, 5, 6, 0]])
```

2. Padding the first dimension:

```python
result = pad_at_dim(t, pad=(2, 2), dim=0, value=-1)
print(result)
```

Output:
```
tensor([[-1, -1, -1],
        [-1, -1, -1],
        [ 1,  2,  3],
        [ 4,  5,  6],
        [-1, -1, -1],
        [-1, -1, -1]])
```

3. Padding the second dimension:

```python
result = pad_at_dim(t, pad=(3, 3), dim=1, value=-2)
print(result)
```

Output:
```
tensor([[-2, -2, -2,  1,  2,  3, -2, -2, -2],
        [-2, -2, -2,  4,  5,  6, -2, -2, -2]])
```

## Additional Tips

1. Use this utility function
