# interpolate_pos_encoding_2d

# Zeta.utils Function: interpolate_pos_encoding_2d

The function `interpolate_pos_encoding_2d` is part of the `zeta.utils` module, and its purpose is to resize a 2D positional encoding to a given target spatial size. The function does this by using bicubic interpolation, which is a method for resampling or interpolating data points on a two-dimensional regular grid.

This function takes in the target spatial size and the positional encoding (pos_embed) as arguments and returns the resized positional encoding.

## Arguments and Return Types

| Arguments              | Type                                                  | Description                                                                                          |
|------------------------|-------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| target_spatial_size    | int                                                   | The desired size for the resized positional encoding.                                                |
| pos_embed              | Tensor                                                | The input positional encoding that needs resizing.                                                   |
                                                                                                                                                       |
| Return                 | Tensor                                                | Returns the positional encoding resized to the given target spatial size.                             |

## Function Definition
```python
def interpolate_pos_encoding_2d(target_spatial_size, pos_embed):
    N = pos_embed.shape[1]
    if N == target_spatial_size:
        return pos_embed
    dim = pos_embed.shape[-1]
    pos_embed, updated = cast_if_src_dtype(pos_embed, torch.bfloat16, torch.float32)
    pos_embed = nn.functional.interpolate(
        pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
            0, 3, 1, 2
        ),
        scale_factor=math.sqrt(target_spatial_size / N),
        mode="bicubic",
    )
    if updated:
        pos_embed, _ = cast_if_src_dtype(pos_embed, torch.float32, torch.bfloat16)
    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return pos_embed
```

## Function Usage and Examples

Here is an example of how to use this function in a general scenario:

Example 1:
```python
import torch
from torch import nn


def cast_if_src_dtype(src, src_dtype, target_dtype):
    if src.dtype == src_dtype:
        return src.to(target_dtype), True
    return src, False


# Creating a random positional encoding
pos_embed = torch.randn(1, 16, 64)  # 2-dimensional, size=(1,16,64)

# Interpolating the positional encoding to a larger spatial size
new_pos_embed = interpolate_pos_encoding_2d(32, pos_embed)
print("Old size:", pos_embed.shape)
print("New size:", new_pos_embed.shape)
```
In this example, an artificial positional encoding of size 1x16x64 is being interpolated to have 32 spatial size, resulting in a new size of 1x1024x64.

## Common Usage Mistakes

One common mistake when using the `interpolate_pos_encoding_2d` function may be not checking the original spatial size of the positional encoding. If a positional encoding has the same spatial size as the target size that you want to resize it to, then the function will return the input positional encoding without resizing.

## References and Further Reading
- [PyTorch nn.functional.interpolate](https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)
- [Resampling or Interpolating](https://en.wikipedia.org/wiki/Resampling_(bitmap))
