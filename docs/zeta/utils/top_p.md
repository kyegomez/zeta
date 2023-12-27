# top_p

# Zeta Utils Library Documentation

The Zeta Utils library is a simple utility library providing a single function, `top_p`, for manipulating and filtering PyTorch tensor-based data sets according to a specified threshold value.

## `top_p` Function

### Function Objective

`top_p` function sorts the values in a tensor, calculates a cumulative sum from a softmax and then applies a threshold to exclude the highest probabilities. Useful when trying to constrain outputs in a certain range.

### Function Definition

```python
def top_p(logits, thres=0.9):
```

### Parameters

| Parameter | Type  | Default Value | Description                                                                                                                               |
|-----------|-------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `logits`  | Tensor| None          | Input tensor containing the values to be processed.                                                                                       |
| `thres`   | Float | 0.9           | Threshold value used to filter the highest probabilities.                                                                                  |


### Return Types

The function returns a Tensor with the same dimensions as the input tensor where the probabilities above the threshold have been filled with negative infinity (`float("-inf")`).

### Internal Functioning 

- First, `logits` are sorted by descending order, receiving both the sorted values and their corresponding indices.
- Next, the softmax of the sorted values is calculated and a cumulative sum over the results is performed.
- Then, a tensor of the same dimension as cum_probs is created, filled with True if the cumulative probability is above the threshold (1 - `thres`), and False otherwise.
- After that, a little shift is made on this tensor to the right so that the values do not exceed the threshold value limit. The first element is explicitly set to 0 (or false).
- Afterwards, the sorted tensor is updated by replacing values at sorted_indices_to_remove (those above threshold) with negative infinity (`float("-inf")`).
- Finally, the `scatter` function rearranges the updated sorted_logits back into the original structure.


## Usage examples 

### Example 1

```python
import torch
from torch.nn import functional as F
from zeta.utils import top_p

logits = torch.randn(10, 10)
result = top_p(logits)
```

This example demonstrates the basic use of the `top_p` function which accepts a tensor with random values and a default threshold value of `0.9`.

### Example 2

```python
import torch
