# l2norm

# Module Name: `l2norm`
---

Function: `l2norm(t, groups=1)`

The `l2norm` is a function written in Python that uses the PyTorch library to normalize tensors. This particular function uses the `L2` or Euclidean norm. The function also handles grouped tensors and normalizes over each group separately. This function can be crucial in many scenarios where input tensors need to be normalized.

## Parameters:

| Parameter | Type | Default value | Description |
|-----------|------|---------------|-------------|
| t         | Tensor | N/A | Input tensor to be normalized. |
| groups    | int | 1 | Number of groups to split the tensor in. |

## Returns:

| Output | Type | Description |
|--------|------|-------------|
| Tensor | Tensor | The L2-normalized tensor.

_Source Code:_

```python
def l2norm(t, groups=1):
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")
```

This function first rearranges the tensor `t` into the specified number of `groups`. After this rearrangement, it normalizes each group using the PyTorch function `F.normalize()` with `p=2`, which indicates the use of L2 or Euclidean norm and `dim=-1`, which normalizes over the last dimension. Finally, the function returns the tensor after rearranging it back to its original structure.

## Usage Examples :

### Example 1:
```python
# Ignore import errors, they are part of the example code
from einops import rearrange
from torch import randn

t = randn(2, 2, 3)
result = l2norm(t, groups=2)
```

In this example, we generate a random tensor `t` with dimensions (2,2,3) using the `torch.randn()` function. Then we call the `l2norm` function with this tensor as the argument and normalize over 2 groups.

### Example 2:
```python
# Ignore import errors, they are part of the example code
from einops import rearrange
from torch import randn

t = randn(3, 3, 3)
result = l2norm(t, groups=1)
```

In this example, we generate a random tensor `t` with dimensions (3,3,3) using the `torch.randn()` function. Then we call the `l2norm` function with this tensor as the argument and normalize over a single group.

### Example 3:
```python
# Ignore import errors, they are part of the example code
from einops import rearrange
from torch import randn

t = randn(4, 4, 2)
result = l2norm(t, groups=4)
```

In this example, we generate a random tensor `t` with dimensions (4,4,2) using the `torch.randn()` function. Then we call the `l2norm` function with this tensor as the argument and normalize over 4 groups.

---

_Tips on usage_:

While using the `l2norm` function, it is necessary to understand the dimensions of the input tensor and the number of groups that we wish to normalize over. More groups would mean more `dim` divisions, followed by individual normalization. This could potentially improve the accuracy of certain ML models where normalization is important.

A suitable value for `groups` would depend entirely on the task at hand and would often need to be determined through experimentation. 

Possible errors may arise if the number of groups is not a divisor of the number of dimensions in the tensor. In such a case, a more suitable value for `groups` should be selected.

---

_For more detailed information, please refer to the Pytorch documentation linked [here](https://pytorch.org/docs/stable/tensors.html) and the Einops documentation linked [here](https://einops.rocks/)_.
