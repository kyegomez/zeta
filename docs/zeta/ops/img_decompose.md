# img_decompose



The `img_decompose` function is designed to decompose a larger batch of images into smaller batches while keeping the individual image dimensions intact. This can be particularly useful when one intends to process the images in smaller groups while maintaining their original resolutions.


### Parameters

`x` (Tensor): The input tensor representing a batch of images. This tensor is expected to have a shape that conforms to the pattern `(batch_size, height, width, channels)`.

### Returns

A tuple representing the shape of the tensor after the `rearrange` operation. It does not return the rearranged tensor but only the shape. The returned shape will always have one extra dimension, splitting the initial batch size into two parts.

## How `img_decompose` Works and Its Usage

`img_decompose` applies the `rearrange` function from the `einops` library on the input tensor `x`, specifying that the batch size (`b1 b2`) will be factored into two separate dimensions, with the first dimension being fixed to `b1=2`. The `rearrange` function is a powerful tool for tensor manipulation, providing a shorthand for expressive operations expressed in Einstein notation.

Below are three different usage examples demonstrating the `img_decompose` function in various scenarios:

### Example 1: Basic Usage

This example shows the basic usage of `img_decompose` to understand how the shape of the input tensor changes.

```python
import torch
from einops import rearrange
from zeta.ops import img_decompose

# Create a dummy tensor representing a batch of 6 images, 
# each image having a height of 32 pixels, width of 32 pixels, and 3 color channels (RGB)
batch_images = torch.randn(6, 32, 32, 3)

# Using img_decompose
new_shape = img_decompose(batch_images)

print("Original shape:", batch_images.shape)
print("New shape after img_decompose:", new_shape)
```

Output:
```
Original shape: torch.Size([6, 32, 32, 3])
New shape after img_decompose: (2, 3, 32, 32, 3)
```

In this example, `img_decompose` processes a tensor representing a batch of 6 images. The function reshapes the batch size from 6 into two dimensions, `2` and `3`, effectively reinterpreting the batch as consisting of 2 smaller mini-batches of 3 images each. The function then returns the shape of the rearranged tensor.

### Example 2: Verifying Output Tensor

In this example, let's show that the `img_decompose` function does not alter the content of the tensor.

```python
import torch
from einops import rearrange
from zeta.ops import img_decompose

# Create a dummy tensor representing a batch of 8 images, 
# each 64x64 pixels with 3 color channels (RGB)
batch_images = torch.randn(8, 64, 64, 3)

# Use img_decompose and reconstruct the tensor from shape
decomposed_shape = img_decompose(batch_images)
reconstructed_tensor = rearrange(batch_images, "(b1 b2) h w c -> b1 b2 h w c", b1=2)

assert reconstructed_tensor.shape == decomposed_shape, "The tensor has not been reconstructed correctly"

print("Original tensor and reconstructed tensor are of the same shape.")
```

Output:
```
Original tensor and reconstructed tensor are of the same shape.
```

In this example, we successfully decompose the input tensor and then reconstruct a tensor with the same shape as indicated by the output of the `img_decompose` function, effectively verifying that the tensor content remains consistent throughout the process.

### Example 3: Practical Application in Data Pipeline

Consider a scenario where we are working with a data pipeline where images come in a batch, but we need to run separate operations on two subsets of this batch. The `img_decompose` function can be used to facilitate this process. 

```python
import torch
from einops import rearrange, repeat
from torchvision import transforms
from zeta.ops import img_decompose

# Function from the zeta.ops library
def img_decompose(x):
    return rearrange(x, "(b1 b2) h w c -> b1 b2 h w c", b1=2).shape

# Data processing pipeline function
def preprocess_and_decompose(batch_images):
    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),        # Resize each image to be 224x224
        transforms.ToTensor(),                # Convert images to tensor format
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for model
    ])
    
    # Assume batch_images is a list of PIL Images
    tensor_images = torch.stack([preprocessing(img) for img in batch_images])

    decomposed_shape = img_decompose(tensor_images)
    decomposed_tensor = rearrange(tensor_images, "(b1 b2) c h w -> b1 b2 c h w", b1=2)
    
    # Now you have two separate batches, which you can process independently
    batch1 = decomposed_tensor[0]
    batch2 = decomposed_tensor[1]
    
    return batch1, batch2

# Mock a batch of 4 PIL images (code for creating these images is omitted for brevity)
batch_images = ...

# Run the preprocessing and decomposition
batch1_processed, batch2_processed = preprocess_and_decompose(batch_images)

# Now, batch1_processed and batch2_processed can be processed by separate pipeline stages or model heads
```

In this scenario, the preprocessing pipeline first converts a batch of PIL Images into a normalized tensor suitable for feeding into a neural network. The `img_decompose` function is then used to obtain the decomposed shape which is used to organize the batch into two subsets. These subsets can then be passed independently through the rest of the pipeline stages.

## Additional Information and Tips

* The function `img_decompose` only returns the shape after rearrangement, not the rearranged tensor itself. If the tensor data is needed in the new shape, you will need to use `rearrange()` and not the `img_decompose()` function.
* The fixed dimension (b1=2) in the `img_decompose` function means that the input tensor's batch size must be an even number to split it evenly. For batch sizes that are not multiples of 2, it's necessary to either adjust the `b1` value or pad the input tensor to fit the specified batch splitting.
* The `img_decompose` function assumes that the input tensor uses the channel last ordering `(batch_size, height, width, channels)`. If a different ordering is used, the `rearrange` pattern would need to be adjusted accordingly.

