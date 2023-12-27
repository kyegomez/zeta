# top_k

# zeta.utils Package Documentation

## The `zeta.utils` module

`zeta.utils` is a utility module that provides various utility functions aimed at simplifying and bolstering the efficiency of data transformation and manipulation processes. This documentation explores, in depth, the usefulness, rationale behind, and significance of the provided functions, which will further help users to leverage them in their specific use cases effectively.

Our focus is the `top_k` function that selectively returns elements from the tensor, having values within the top k percentile.

<br>

# Function Name: `top_k`

The `top_k` function is aimed at aiding common procedures encountered in machine learning and data science involving tensor manipulations. Specifically, it speeds up the rank-based filtering of elements in a tensor. 

**Definition/Signature**:

```python
def top_k(logits, thres=0.9):
```

**Parameters**:

The function accepts the following arguments:

| Parameters | Type   | Description                                                                                              | Default Value |
|------------|--------|----------------------------------------------------------------------------------------------------------|---------------|
| logits     | tensor | A tensor whose elements are required to be ranked and top k percentile to be separated.  | None          |
| thres      | float  | A threshold value determining the percentile of top elements to be selected from the tensor. | 0.9           |

<br>

**How It Works**:

The `top_k` function works by utilizing PyTorch's topk function to pull the top-k elements from a tensor, based on the specified threshold. It then builds a new tensor filled with -inf (representing negative infinity) and scatter the top-k elements into it. This implies that the returned tensor has the top-k elements from the original tensor and -inf for the rest. This aids easy selection and corresponding actions on the top-k elements without the strain of performing an explicit sort operation on the tensor and then slicing off the top-k elements.

**Returns**:

A tensor which has the top-k elements from the original tensor and -inf for the rest.

<br>

**Example Usage(s)**:

Below are three illustrative examples of leveraging the `top_k` function:

**Example 1:**

```python
import torch
from math import ceil
from zeta.utils import top_k

# Initialize tensor
tensor = torch.rand(1, 10) 

# Apply function with threshold 0.9
filtered_tensor = top_k(tensor, thres=0.
