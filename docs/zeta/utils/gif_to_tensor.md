# gif_to_tensor

# Module/Function Name: gif_to_tensor

## Introduction

The `gif_to_tensor` function in the `zeta.utils` library is a utility function to convert an animated GIF into a PyTorch tensor. This function is very handy when handling image data, especially when the task is related to processing animated GIFs in machine learning or deep learning applications. 

In the `zeta.utils` library, the `gif_to_tensor` function serves as an essential bridge between raw GIF files and the tensor format required for many other PyTorch operations. 

## Function Definition

```python
def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, chanels=channels)))
    return torch.stack(tensors, dim=1)
```

## Parameters

| Parameter   | Type                               | Description                                                                                                                               | Default Value         |
|-------------|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| `path`        | str                                | A string specifying the path to the gif file.                                                                                              | None                  |
| `channels`    | int                                | An integer specifying the number of channels in the image. Typical values are 1 (grayscale), 3 (RGB), or 4 (RGBA).                        | 3 (RGB)               |
| `transform`   | torchvision.transforms.Transforms | A PyTorch transformation to be applied to each image frame. PyTorch provides a number of transformations like `ToTensor()`, `Normalize()`. | `T.ToTensor()` |

## Functionality and Usage

This function performs the following operations:

1. Opens the GIF image using the path provided.
2. Iterates over all the frames in the GIF image.
3. Applies the transformation to each frame to convert it into a PyTorch tensor.
4. Stacks all the tensors for each frame along a new dimension.

The output of the function is a single tensor representing all frames of the GIF. The dimension corresponding to the frames in the output tensor is 1.

Below, we show three examples of using this function:

1. **Basic Usage:**
    In this simplest use case, we only need to provide the path to the GIF file. The function will return a tensor representing the GIF, using default settings for channels (RGB) and transformation (convert to tensor).

    ```python
    import torchvision.transforms as T
   
