# img_compose_bw


The primary role of `img_compose_bw` is to rearrange the dimensions of a 4D tensor representing a batch of black and white images so that all the images in the batch are concatenated horizontally, resulting in a single wide image composed of the batch. This utility can be particularly useful for visualization purposes or for operations where it's advantageous to view the entire batch as one wide image strip.

### Parameters

| Parameter | Type | Description |
| ----------| ---- | ----------- |
| `x`       | Tensor | A 4D tensor with dimensions `(b, h, w, c)` where `b` is the batch size, `h` is the height, `w` is the width, and `c` is the number of channels (should be 1 for black and white images). |

### Returns

| Return    | Type  | Description |
| ----------| ------| ----------- |
| `tensor`  | Tensor | A rearranged 3D tensor with dimensions `(h, b * w, c)`. |

## Functionality and Usage

The `img_compose_bw` function uses the `rearrange` operation, commonly associated with a library named `einops`. This operation allows complex tensor transformations with a concise and readable syntax.

The purpose of the function is to take a batch of black and white images in the form of a 4D tensor `(batch, height, width, channels)` and transform it into a 3D tensor where images are concatenated horizontally across the width.

### Example Usage:

Before diving into the examples, let's clarify the necessary imports and prerequisites expected to run the following code.

Imports and setup.

```python
# Note: This assumes that einops is installed in your environment.
import torch

from zeta.ops import img_compose_bw
```

#### Example 1: Basic Usage

```python
# Assuming you have a batch of 4 black and white images,
# each of dimensions 64x64 pixels (1 channel for B&W images)
batch_size = 4
height = 64
width = 64
channels = 1  # Channels are 1 for B&W images

# Create a dummy batch of images
batch_images = torch.rand(batch_size, height, width, channels)

# Use img_compose_bw to rearrange the batch into a single wide image
wide_image = img_compose_bw(batch_images)

# wide_image now has the shape: (64, 256, 1)
print(wide_image.shape)
```

#### Example 2: Visualization

One common reason to use `img_compose_bw` is to prepare a batch of images for visualization.

```python
import matplotlib.pyplot as plt

# Visualize the result
plt.imshow(
    wide_image.squeeze(), cmap="gray"
)  # Remove the channel dimension for plotting
plt.axis("off")  # Hide the axes
plt.show()
```

#### Example 3: Processing before passing to a model

You might want to preprocess your image batch before passing it through a convolutional neural network (CNN).

```python
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1
        )
        # More layers here...

    def forward(self, x):
        x = self.conv1(x)
        # More operations...
        return x


# Instantiate the model
model = SimpleCNN()

# Wide_image is already a tensor of shape (height, width*batch_size, channels)
# Reshape it to (channels, height, width*batch_size) to match the expected input format of PyTorch CNNs
wide_image_cnn = wide_image.permute(2, 0, 1).unsqueeze(0)  # Adds a batch dimension

# Pass the tensor through the CNN
output = model(wide_image_cnn)

print(output.shape)
```

Multiple examples demonstrate the adaptability of `img_compose_bw` to different tasks. Users can easily integrate this function into their image processing pipelines when working with batches of black and white images.

## Additional Information and Tips

1. The `img_compose_bw` function specifically works with black and white images, represented by a single channel. If using this function on RGB images, ensure that the color channels are properly handled before applying the function.

2. The function assumes that the input tensor layout is `(batch, height, width, channels)`. If your tensors are structured differently, you might need to permute the dimensions to match this format.

3. The `img_compose_bw` function can be easily modified to concatenate images vertically or in any other custom layout by changing the pattern string passed to the `rearrange` function.

## Conclusion

In this documentation, we explored the `img_compose_bw` function from our  `zeta.ops` library, intended for the transformation of image tensors for black and white images. We reviewed the function definition, parameters, usage examples, and additional tips to ensure effective application of the function in various scenarios.

This utility serves as a convenient tool for visualizing and processing batches of black and white images, fitting seamlessly into the preprocessing pipelines of image-related machine learning tasks.

