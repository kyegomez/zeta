# l2norm

# Module Name: zeta.utils

## Function: l2norm
```python
def l2norm(t, groups=1):
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")
```

### Overview
The function `l2norm` as the name suggests, is used for L2 normalization of tensors. L2 normalization is the process of dividing a feature vector by its L2 norm, which results in a vector on the unit sphere. It helps deal with issues involving scale variance in data.

The `l2norm` function takes in a tensor and an optional `groups` parameter, rearranges the elements of the tensor as per the `groups` parameter, performs the normalization and then again rearranges elements to their original order.

The function makes use of the `rearrange` function from the `einops` library and the `normalize` function from PyTorch's `torch.nn.functional` library.

### Parameters
The `l2norm` function has the following parameters:

| Argument | Type | Description | Default Value |
| --- | --- | ---| --- |
| t | torch.Tensor | The tensor that requires L2 normalization. | - |
| groups | int | The number of groups to divide the tensor into before applying normalization. | 1 |

### Usage
Here are three examples showcasing the usage of the `l2norm` function:

#### Example 1 
```python
from zeta.utils import l2norm
import torch

# Creating a 3-dimensional tensor
tensor = torch.rand(4,2,2)

# Using l2norm without specifying groups
normalized_tensor = l2norm(tensor)

# Print the output
print(normalized_tensor)
```

In this example, we create a random 3-dimensional tensor and use the `l2norm` function to normalize it without specifying the `groups` parameter. Thus, the tensor will not be divided into groups before normalization.

#### Example 2
```python
from zeta.utils import l2norm
import torch

# Creating a 3-dimensional tensor
tensor = torch.rand(4,2,2)

# Using l2norm specifying groups as 2
normalized_tensor = l2norm(tensor, groups=2)

# Print the output

