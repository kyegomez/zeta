# sparsemax

`sparsemax` offers an alternative to the traditional softmax function, commonly used in classification tasks and attention mechanisms within neural networks. It is designed to produce sparse probability distributions, which can be useful for interpretability and models where only a few items should have substantial weight.

### Functionality
The `sparsemax` function transforms an input tensor into a sparse probability distribution. It operates by sorting its input in descending order and then applying a thresholding function to decide the set of selected logits.

The operation can be summarized as:

`sparsemax(z) = max(0, z - tau(z))`

Here, `tau(z)` represents a threshold that is determined by the sum of the largest-k logits, scaled by k:

`tau(z) = (sum_i=1^k z_i - 1) / k`

where `z` is the input tensor and `k` is a user-specified number representing the number of elements to keep.

### Usage
The `sparsemax` is used much like softmax when you need to pick only the top k logits to focus on, pushing the rest towards zero in the output distribution.

### Parameters

| Parameter | Type        | Description                                            |
|-----------|-------------|--------------------------------------------------------|
| x         | Tensor      | The input tensor upon which to apply sparsemax.        |
| k         | int         | The number of elements to keep in the sparsemax output.|

### Examples

#### Example 1: Basic Usage

```python
import torch
from zeta.ops import sparsemax

# Initialize an input tensor
x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

# Apply sparsemax, keeping the top 3 elements
k = 3
output = sparsemax(x, k)

print(output)
```

#### Example 2: Large Tensors

```python
import torch
from zeta.ops import sparsemax

# Initialize a large tensor with random values
x = torch.randn(10, 1000)

# Applying sparsemax, selecting top 50 elements
k = 50
output = sparsemax(x, k)

print(output)
```

#### Example 3: Error Handling

```python
import torch
from zeta.ops import sparsemax

try:
    # Initialize an input tensor
    x = torch.tensor([[1.0, 2.0, 3.0]])

    # Try to apply sparsemax with an invalid k
    k = 5 # More than the number of logits
    output = sparsemax(x, k)
except ValueError as e:
    print(e)
```

### Notes on Implementation
The internal implementation of `sparsemax` considers edge cases, such as when `k` is greater than the number of logits, or where the practical value of `k` needs to be adjusted. They are clarified through error messages and internal adjustments within the function.

### Additional Information

The `sparsemax` function is part of the `zeta.ops` library which focuses on providing operations that are useful for structured and sparse outputs in neural networks. These functions are designed to be efficient and differentiable, which makes them suitable for use in gradient-based learning methods. 

### References
- [André F. T. Martins, Ramón Fernandez Astudillo. "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification." (2016)](https://arxiv.org/abs/1602.02068)
- PyTorch Documentation: [torch.Tensor](https://pytorch.org/docs/stable/tensors.html)

For further exploration of the `sparsemax`, or additional utility functions within the `zeta.ops` library, users may refer to the official documentation or reach out to the community forums for discussions and support.

---

