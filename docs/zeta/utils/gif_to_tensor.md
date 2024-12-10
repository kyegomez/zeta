# gif_to_tensor

# Module Name: `gif_to_tensor`

The `gif_to_tensor` module is a Python function that converts a GIF (Graphics Interchange Format) image into a tensor. This module is very useful in machine learning tasks where GIFs are used as input. For instance, in video understanding or some forms of anomaly detection, short snippets of video as GIFs can be very useful. Hence this function is a fundamental and powerful function that can work with the Pytorch framework in creating machine learning models.

## Function Definition

``` python
def gif_to_tensor(path: str, channels: int = 3, transform = torch.transforms.ToTensor()) -> torch.Tensor:
    """
    This function reads a GIF image from disk, applies transforms and converts it into a stack of tensors.

    Parameters:

    - path (str): The file path of the GIF image.
    - channels (int): The number of color channels in the image. Default value is 3 (RGB). 
    - transform (torch.transforms.ToTensor()): The transform function that is applied to each frame of the GIF image. Default transform is ToTensor() which converts the image into tensor.

    Returns:

    - torch.Tensor: A tensor representation of the GIF image.

    Note:

    - The created tensor is a 4D-tensor of shape (frames, channels, height, width) where frames is the number of frames in the GIF image.
    """

    # function implementation here
```

## Function Usage
The `gif_to_tensor` function is fairly simple and straightforward to use. It takes three parameters - `path`, `channels` and `transform`- and returns a tensor. You primarily need to provide the `path` parameter - which points to the GIF image you want to convert into a tensor, while the other parameters are optional.

Here are three ways of using the `gif_to_tensor` function:

``` python
import torch
import torchvision.transforms as T
from PIL import Image

# gif_to_tensor function
def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, chanels=channels)))
    return torch.stack(tensors, dim=1)

# Example 1: Basic usage with just the path parameter
result = gif_to_tensor('./path_to_your_gif.gif')
print(result.shape)  # Outputs: torch.Size([Frames, 3, Height, Width])

# Example 2: Specifying the number of channels
result = gif_to_tensor('./path_to_your_gif.gif', channels=1)
print(result.shape)  # If the input gif is grayscale, Outputs: torch.Size([Frames, 1, Height, Width])

# Example 3: Applying multiple transforms
custom_transform = T.Compose([T.Resize((100, 100)), T.ToTensor()])
result = gif_to_tensor('./path_to_your_gif.gif', transform=custom_transform)
print(result.shape)  # Outputs: torch.Size([Frames, 3, 100, 100]), if the input gif has 3 color channels
```

## Additional Information
The created tensor is a 4D tensor of shape (frames, channels, height, width), where frames is the number of frames in the gif image. The values (pixel intensities) in the returned tensor are in the range `[0, 1]` if the transform `T.ToTensor()` is used.

Notice that the `seek_all_images` function used in the implementation of `gif_to_tensor` is not defined in the provided code. This function is expected to find and return all frames in the animated gif image. You need to consider this when using `gif_to_tensor` in your code. Make sure to define such a function or use equivalent functionality from existing libraries.

## References
For more information on torch.Tensor, PIL.Image and torchvision.transforms, refer to:
- Pytorch's official documentation: [torch.Tensor](https://pytorch.org/docs/stable/tensors.html)
- Python Imaging Library (PIL) documentation: [PIL.Image](https://pillow.readthedocs.io/en/stable/reference/Image.html)
- Torchvision transforms documentation: [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)
