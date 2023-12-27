# video_tensor_to_gift

# Module Name: zeta.utils

## Function: video_tensor_to_gift

    ```
    This function converts a tensor representation of a video into a GIF file.
    It takes a tensor video as input, unbinds the tensor, converts each image-like tensor in the video to a PIL image,
    and then saves all these images in a GIF file.

    Parameters:
    - tensor (tensor): A tensor containing the video data.
    - path (str): The path where the GIF should be saved.
    - duration (int): The time (in milliseconds) that each frame should be displayed. Default: 120 ms.
    - loop (int): The number of times the GIF should loop. 
                  0 for infinite loop, and other integer values for specific count of loops. Default: 0 (infinite loop).
    - optimize (bool): If True, the resulting GIF will be optimized to save space.
                       Optimization can take more time and result in minimal changes, so if you’re in a hurry, or don’t care about file size, you can skip optimization. Default: True.

    Returns:
    list: list of images created from the tensors.
    ```
```python
def video_tensor_to_gift(tensor, path, duration=120, loop=0, optimize=True):
    images = map(T.ToPilImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(
        path,
        save_all=True,
        append_images=rest_imgs,
        duration=duration,
        loop=loop,
        optimize=optimize,
    )
    return images
```

## Usage Examples:

### Example 1:

```python
# import the necessary libraries
import torch
from torchvision import transforms as T
from zeta.utils import video_tensor_to_gift

# Define a tensor for generating a video:
video_data = torch.rand(10, 10, 3, 64, 64)

# Call the function:
video_tensor_to_gift(video_data, 'test.gif')
```
In this example, we generate a tensor of random pixel intensity values. The generated GIF file will be saved in the current working directory with the name 'test.gif'. The gif file be looping indefinitely.

### Example 2: 

```python
# import the necessary libraries
import torch
from torchvision import transforms as T
from zeta.utils import video_tensor_to_gift

# Define a tensor for
