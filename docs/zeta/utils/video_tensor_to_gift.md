# video_tensor_to_gift

# Module Name: zeta.utils

## Function: video_tensor_to_gift

```python
def video_tensor_to_gift(tensor, path, duration=120, loop=0, optimize=True):
    """
    This function converts a video tensor into a gif and then saves it on the provided path.

    Parameters:
    - tensor (tensor): A tensor representing a video. The tensor should be 5-dimensional (B, T, C, H, W).
    - path (str): The location and filename where the gif should be saved. Built-in gif extension is recommended to ensure correct file format.
    - duration (int): The duration for which each frame should be displayed before transitioning to the next. Default is 120 (in milliseconds).
    - loop (int): The number of times the gif should loop. A value of 0 means the gif will loop indefinitely. Default is 0.
    - optimize (bool): A flag specifying whether the gif should be optimized. If set to True, the gif would have smaller size at the cost of quality. Default is True.

    Returns:
    - images: A sequence of images that constitute the gif.

    Examples:

    This is a simple usage case.
    
    ```python
    from torchvision.transforms import functional as T
    import torch
    from zeta.utils import video_tensor_to_gift

    # Generate a random tensor representing a video
    tensor = torch.rand(1, 10, 3, 64, 64)

    # Convert tensor to gif and save
    path = "./random_video.gif"
    video_tensor_to_gift(tensor, path)
    ```

    This example showcases usage with different arguments.
    
    ```python
    from torchvision.transforms import functional as T
    import torch
    from zeta.utils import video_tensor_to_gift

    # Generate a random tensor representing a video
    tensor = torch.rand(1, 10, 3, 64, 64)

    # Convert tensor to gif and save with custom duration, loop, and optimization set.
    path = "./random_video.gif"
    video_tensor_to_gift(tensor, path, duration=200, loop=1, optimize=False)
    ```

    """
    images = map(T.ToPilImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(
        path,
        save_all=True,
        appeqnd_images=rest_imgs,
        duration=duration,
        loop=loop,
        optimize=optimize,
    )
    return images
```

## Architecture

The function `video_tensor_to_gift` works by first unbinding the video tensor along the time dimension using the `unbind()` function, which returns a tuple of all slices along that dimension. This breaks the tensor into a sequence of image tensors.

The `map()` function is then used to apply `T.ToPilImage()`, a torchvision functional transform, to each of these image tensors. This converts each tensor into a PIL Image.

The sequence of PIL Images is then split, with the `first_img` separated from the `rest_imgs`. 

The function then uses the `first_img.save()` method to save all the images as a gif at the provided path. The `save_all` parameter set to `True` signals that all images should be saved in the gif, not just the first one. The `append_images` parameter specifies the additional images to be added, which in this case are the rest of the images. The `duration`, `loop`, and `optimize` parameters control the behavior of the gif.

### Note:
Optimizing the gif can reduce the size of the gif file but may also slightly degrade the image quality.

This function is handy for quick visualization and debugging purposes, as it can help analyze the content of video tensors during model development.

### References and further resources:

For understanding more about the image saving process in PIL:
https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif 

For understanding more about TorchVision transform functions:
https://pytorch.org/vision/stable/transforms.html 

For more details on PyTorch tensor functions such as `unbind`:
https://pytorch.org/docs/stable/tensors.html 
