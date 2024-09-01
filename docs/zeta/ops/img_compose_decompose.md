# img_compose_decompose

Function `img_compose_decompose` restructures a batch of images by decomposing each image into sub-images and then composing a new set of "images" by arranging these sub-images.

This transformation function is useful when working with tasks that involve image-to-image translation where sub-images need to be rearranged, such as styling certain quadrants of images differently, or when data needs to be preprocessed for multi-scale feature extraction.

## Overview and Introduction

The `img_compose_decompose` function comes from the `zeta.ops` library (), which provides utilities to manipulate multidimensional data, specifically tailored for image data in this case. This library is designed to simplify the preprocessing and augmentation operations that are often required in computer vision tasks.

## Function Definition

Below is the definition of the `img_compose_decompose` function:

```python
def img_compose_decompose(x):
    """
    Rearranges a batch of images by decomposing each image into sub-images and then composes a new set of "images" by arranging these sub-images.

    Parameters:
    - x (Tensor): A batch of images with shape (b, h, w, c), where `b` is the total batch size, `h` and `w` are the height and width of each image, and `c` is the number of channels.
    """
    return rearrange(x, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=2)
```

The function assumes that the input tensor `x` is of shape `(b, h, w, c)` and utilizes the `rearrange` function from the `einops` library to perform the restructuring.

### Parameters

| Parameter | Type  | Description                                                             | Default |
|:----------|:------|:------------------------------------------------------------------------|:--------|
| x         | Tensor| A batch of images with shape `(b, h, w, c)`                              | None    |

## Functionality and Usage

The `img_compose_decompose` function works by decomposing each image in the batch into 2x2 sub-images and then arranging them in a grid to create a new set of composed images. The new image dimensions become `(2*h, 2*w, c)`, effectively composing images that are 4 times larger in the number of pixels.

### Usage Examples

#### Example 1: Basic Usage

```python
import torch

from zeta.ops import img_compose_decompose

# Assume x has a shape of (4, 100, 100, 3), representing 4 images of 100x100 pixels with 3 color channels
x = torch.randn(4, 100, 100, 3)

# Decompose and compose the images
result = img_compose_decompose(x)

# Resulting tensor shape: (2*100, 2*100, 3)
print(result.shape)  # should output torch.Size([200, 200, 3])
```

#### Example 2: Working with a DataLoader

```python
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from zeta.ops import img_compose_decompose

# Load CIFAR10 images
cifar10_dataset = CIFAR10(".", train=True, download=True, transform=ToTensor())
cifar10_loader = DataLoader(cifar10_dataset, batch_size=8, shuffle=True)

# Iterate over the data loader
for batch, (images, labels) in enumerate(cifar10_loader):
    # Apply img_compose_decompose function to the batch of images
    composed_images = img_compose_decompose(images)
    # Process composed images further
    # ...
    break  # Processing just one batch for demonstration
```

#### Example 3: Visualizing the Transformation

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from zeta.ops import img_compose_decompose

# Load an image
image = Image.open("sample_image.jpg")
image_np = np.array(image)

# Add batch and channel dimensions to the image
image_batch = image_np.reshape(1, *image_np.shape)

# Apply the img_compose_decompose function
composed_image = img_compose_decompose(image_batch)

# Show the original and the composed images
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(composed_image[0])
plt.title("Composed Image")

plt.show()
```

## Additional Information and Tips

- The `img_compose_decompose` function currently works with a fixed number of sub-images (2x2). For different configurations, modifications to the function or the `rearrange` pattern will be necessary.
- The function is built on top of the `einops.rearrange` function, which is a versatile tool for tensor manipulation. Users unfamiliar with `einops` may benefit from reading its documentation for a deeper understanding of tensor operations.

## References and Resources

- For more information on the `einops.rearrange` function, please refer to the [einops documentation](https://einops.rocks/).
- Users seeking to apply this function to deep learning models might consider reading about PyTorch's `Dataset` and `DataLoader` classes in the [PyTorch documentation](https://pytorch.org/docs/stable/data.html).
