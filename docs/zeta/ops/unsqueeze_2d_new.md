# `unsqueeze_2d_new` Function Documentation

The `unsqueeze_2d_new` is a custom function within the `zeta.ops` library which performs a specific operation onto input tensors, notably rearranging and scaling the spatial dimensions. The following extensive documentation will cover the purpose, architecture, working principle, and usage examples of this function.

---

## Overview and Introduction

The `unsqueeze_2d_new` function serves as a utility within deep learning operations, specifically those that involve manipulating the spatial dimensions of tensors, typically within the context of convolutional neural networks (CNNs) or other architectures dealing with image or grid-like data. The function's main purpose is to expand the spatial dimensions (height and width) of the input tensor by a specified scaling factor. This is akin to performing an 'un-squeeze' operation in two dimensions, enabling finer spatial resolution processing or preparing the tensor for upscaling operations.

## Function Definition

```python
def unsqueeze_2d_new(input, factor=2):
    """
    Expands the spatial dimensions of an input tensor by rearranging its elements according to a given spatial factor.

    Parameters:
    - input (Tensor): A 4D input tensor with shape (batch_size, channels, height, width).
    - factor (int): The scaling factor for the spatial dimensions. Default value is 2.

    Returns:
    - Tensor: A tensor with expanded spatial dimensions.
    """
    return rearrange(
        input, "b (c h2 w2) h w -> b c (h h2) (w w2)", h2=factor, w2=factor
    )
```

**Parameters and Return Value:**

| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| `input`   | Tensor | A 4D input tensor with dimensions representing batch size, number of channels, height, and width, respectively. | None (required) |
| `factor`  | int | The scaling factor by which to expand the spatial dimensions of the input tensor: `height` and `width`. | 2 |

| Return Value | Type | Description |
|--------------|------|-------------|
| (Unnamed)    | Tensor | The output tensor after spatial dimension expansion, having larger height and width by a factor of `factor`. |

## Detailed Explanation and Usage

### How It Works

The `unsqueeze_2d_new` utilizes the `rearrange` function from the `einops` library or a similar tensor manipulation library, which allows for a concise and readable tensor transformation. The operation performed by `unsqueeze_2d_new` implicitly reshapes and expands the 2D spatial dimensions (`height` and `width`) without altering the data within the batch and channel dimensions. This operation is useful in neural networks where a change in spatial resolution is required, such as in generative networks, spatial attention mechanisms, and feature pyramids.


### Usage Example 1: Basic Usage

This example demonstrates how to use the `unsqueeze_2d_new` function to double the height and width of a random tensor.

```python
import torch
from zeta.ops import unsqueeze_2d_new

# 1. Prepare a random tensor with shape (batch_size=1, channels=3, height=4, width=4)
input_tensor = torch.rand(1, 3, 4, 4)

# 2. Apply the unsqueeze_2d_new function with the default factor
output_tensor = unsqueeze_2d_new(input_tensor)

# 3. Verify the shape of the output tensor
assert output_tensor.shape == (1, 3, 8, 8)
```

### Usage Example 2: Custom Scaling Factor

In this example, we show how to use a different scaling factor to alter the spatial scaling performed by the function.

```python
import torch
from zeta.ops import unsqueeze_2d_new


# 1. Prepare a random tensor with shape (batch_size=1, channels=3, height=4, width=4)
input_tensor = torch.rand(1, 3, 4, 4)

# 2. Apply the unsqueeze_2d_new function with a custom factor of 3
output_tensor = unsqueeze_2d_new(input_tensor, factor=3)

# 3. Verify the shape of the output tensor
assert output_tensor.shape == (1, 3, 12, 12)
```

### Usage Example 3: Integrating into a Neural Network Layer

Lastly, we will demonstrate how `unsqueeze_2d_new` can be integrated into a  neural network model layer. This could be part of an up-sampling process within a generative model:

```python
import torch
import torch.nn as nn
from zeta.ops import unsqueeze_2d_new


class UpsampleLayer(nn.Module):
    def __init__(self, factor=2):
        super(UpsampleLayer, self).__init__()
        self.factor = factor

    def forward(self, x):
        return unsqueeze_2d_new(x, factor=self.factor)


# Model instantiation and usage
upsample_layer = UpsampleLayer(factor=2)
input_tensor = torch.rand(1, 3, 4, 4)
output_tensor = upsample_layer(input_tensor)

assert output_tensor.shape == (1, 3, 8, 8)
```

---

## Additional Information and Tips

The `unsqueeze_2d_new` function is highly dependent on the `rearrange` operation and thus, relies on the functionality provided by the `einops` library. When different tensor shapes or patterns are needed, the pattern string inside the `rearrange` function would need to be adapted accordingly, making this utility highly customizable.

Be mindful that increasing the spatial dimensions can significantly increase the memory usage, especially when dealing with large tensors. Therefore, ensure that your hardware is capable of handling the larger tensor sizes that may result from using this function within your models.

## References and Further Reading

For further details on tensor operations and customization options available with the `einops` library or similar tensor manipulation libraries, consider the following resources:

- Einops documentation and guides: [https://einops.rocks/](https://einops.rocks/)
- Official PyTorch documentation on tensor operations: [https://pytorch.org/docs/stable/tensors.html](https://pytorch.org/docs/stable/tensors.html)

This documentation has provided an in-depth look at the `unsqueeze_2d_new` function, its architecture, functionality, and examples of usage within the scope of tensor manipulation for machine learning and deep learning applications.
