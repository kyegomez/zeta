# img_order_of_axes

The `img_order_of_axes` function is a utility designed to reorder the axes of an image tensor for processing or visualization purposes. Its primary use case is to transform a batch of images with the format batch-height-width-channel (b, h, w, c) into a format suitable for displaying multiple images in a single row, maintaining the channel order.

This documentation provides an in-depth understanding of the `img_order_of_axes` function, its architecture, and the rationale behind its design. We will cover multiple usage examples, detailing the parameters, expected inputs and outputs, along with additional tips and resources.

The `img_order_of_axes` function plays a crucial role in scenarios where a batch of images needs to be combined into a single image with individual images laid out horizontally. This function is particularly useful when there is a need to visualize multiple similar images side by side, such as comparing different stages of image processing or visualization of input-output pairs in machine learning tasks.

## Function Definition

### img_order_of_axes(x)
Rearranges the axes of an image tensor from batch-height-width-channel order to height-(batch * width)-channel order.

#### Parameters:

| Parameter | Type        | Description |
|-----------|-------------|-------------|
| x         | Tensor      | A 4-dimensional tensor representing a batch of images with shape (b, h, w, c), where b is the batch size, h is the height, w is the width, and c is the number of channels. |

#### Returns:
A rearranged tensor that combines the batch and width dimensions, resulting in a shape of (h, b * w, c).


### Usage Example 1:

Visualizing a batch of images side by side:

```python
import torch
from einops import rearrange
from zeta.ops import img_order_of_axes

# Create a dummy batch of images with shape (b, h, w, c)
batch_size, height, width, channels = 4, 100, 100, 3
dummy_images = torch.rand(batch_size, height, width, channels)

# Use `img_order_of_axes` to prepare the tensor for visualization
reordered_images = img_order_of_axes(dummy_images)

# `reordered_images` will have the shape (height, batch_size * width, channels)
print(reordered_images.shape)  # Expected output (100, 400, 3)
```

### Usage Example 2:

Comparing image pairs before and after processing:

```python
import torch
from einops import rearrange
from zeta.ops import img_order_of_axes

# Create a dummy batch of original images and processed images
batch_size, height, width, channels = 2, 100, 100, 3
original_images = torch.rand(batch_size, height, width, channels)
processed_images = torch.rand(batch_size, height, width, channels)

# Concatenate the original and processed images in the batch dimension
combined_batch = torch.cat((original_images, processed_images), dim=0)

# Reorder the axes for side by side comparison
comparison_image = img_order_of_axes(combined_batch)

# Visualize or save `comparison_image` as needed
```

### Usage Example 3:

Preparing a batch of images for a single forward pass in a convolutional neural network (CNN):

```python
import torch
from einops import rearrange
from zeta.ops import img_order_of_axes

# Assuming `model` is a pre-defined CNN that expects input of shape (h, w, c)
batch_size, height, width, channels = 8, 64, 64, 3
input_images = torch.rand(batch_size, height, width, channels)

# Combine all images side by side to form a single large image
large_image = img_order_of_axes(input_images)

# Now `large_image` can be fed into the CNN as a single input
output = model(large_image.unsqueeze(0))  # Add batch dimension of 1 at the beginning
```

## Additional Information and Tips

- It's important to note that the `rearrange` function used within `img_order_of_axes` is not a PyTorch built-in function. It requires the `einops` library which offers more flexible operations for tensor manipulation.
- To install `einops`, use the package manager of your choice, e.g., `pip install einops` for Python's pip package manager.
- When visualizing the rearranged tensor, ensure that the visualization tool or library you choose can handle non-standard image shapes, as the resulting tensor will have a width that is a multiple of the original width.
