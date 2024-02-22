# img_width_to_height


Welcome to the *zeta.ops* library documentation, where we delve into the intuitive and powerful operation `img_width_to_height`. This documentation will serve as a comprehensive guide to understanding the function's architecture, usage, and purpose with in-depth examples and explicit instructional content. The `img_width_to_height` function is designed to reshape image tensor dimensions for various purposes such as algorithmic preprocessing or network input formatting.

The *zeta.ops* library, although , remains essential for transformations and operations on multi-dimensional data where the shape of the tensor is paramount to the downstream application. The `img_width_to_height` function reorganizes a 4D tensor typically used for batched image data, adjusting its spatial orientation by altering the width and height dimensions.

Before we proceed, ensure you possess a basic understanding of PyTorch, as the function manipulates PyTorch tensors and uses the `rearrange` function from the `einops` library for tensor operations.

## img_width_to_height Function Definition

```python
def img_width_to_height(x):
    return rearrange(x, "b h (w w2) c -> (h w2) (b w) c", w2=2)
```

`img_width_to_height` is a function that accepts a single argument `x`, which represents a 4D tensor typically containing image data in batch.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| x         | Tensor | A 4D PyTorch tensor with shape `(b, h, w, c)` where `b` is the batch size, `h` is the height, `w` is the width, and `c` is the channel depth of the image data. |

### Returns

| Return    | Type | Description |
|-----------|------|-------------|
| Tensor | Tensor | A rearranged 4D PyTorch tensor with a new shape `(h w2, b w, c)` where `w2` is hardcoded to be 2 within the scope of this function. |

### Functionality and Usage

#### Why this Architecture?

The architecture of `img_width_to_height` provides a convenient way to group spatial dimensions of images in preparation for certain types of neural network layers that require specific input shapes or for image preprocessing tasks that benefit from a reshaped tensor.

Its reliance on `einops.rearrange` allows for flexible and readable tensor transformation, which is essential when working with multi-dimensional data.

#### How it Works

The `rearrange` method from the `einops` library uses a string-based mini-language for tensor operations. In this instance, the following pattern is used: `"b h (w w2) c -> (h w2) (b w) c"`. This pattern means the input tensor `x` is treated as having batch (`b`), height (`h`), width (`w` times a width factor `w2`), and channels (`c`). It then reshapes the tensor into a new shape were height is multiplied by `w2`, the batch size is multiplied by the original width and the channel remains the same.

#### Usage Examples

**Example 1: Basic usage of img_width_to_height**

```python
import torch
from einops import rearrange

from zeta.ops import img_width_to_height

# Initialize a dummy 4D tensor representing two RGB images (batch size: 2, width: 4, height: 3, channels: 3)
batched_images = torch.randn(2, 3, 4, 3)

# Use our function to transform the tensor's shape
transformed_images = img_width_to_height(batched_images)

print(transformed_images.shape)  # Output -> torch.Size([6, 8, 3])
```

**Example 2: Visualizing the transformation**

```python
import matplotlib.pyplot as plt

# Display original image tensors
fig, axes = plt.subplots(1, 2)
for i, img_tensor in enumerate(batched_images):
    axes[i].imshow(img_tensor.permute(1, 2, 0))
    axes[i].set_title(f"Original Image {i+1}")
plt.show()

# Display transformed image tensors
transformed_shape = transformed_images.shape
for i in range(transformed_shape[1] // transformed_shape[0]):
    img_tensor = transformed_images[:, i : i + transformed_shape[0], :]
    plt.imshow(img_tensor.permute(1, 0, 2))
    plt.title(f"Transformed Image {i+1}")
    plt.show()
```

**Example 3: Preparing tensor for a custom convolutional layer**

```python
import torch.nn as nn


class CustomConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=(3, 3))

    def forward(self, x):
        x = img_width_to_height(x)
        # Assuming that the custom convolutional layer expects a single channel input
        x = x.unsqueeze(1)  # Add a channel dimension
        output = self.conv(x)
        return output


# Initialize model and dummy input
model = CustomConvLayer()
input_tensor = torch.randn(2, 3, 4, 3)  # (batch, height, width, channels)

# Forward pass
output = model(input_tensor)

print(output.shape)  # Output size will depend on the convolutional layer properties
```

### Additional Information and Tips

- Make sure that the input tensor `x` has the width dimension to be an even number. The function assumes a division by 2 for width (`w2=2`).
- Consider padäding your image tensor to an even width if it's odd-sized before using this function.
- `einops.rearrange` adds a significant level of readable abstraction for tensor reshaping, but you should familiarize yourself with its mini-language to make the most out of it.

