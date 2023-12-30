# merge_small_dims

allows reshaping of a tensor by merging its smaller dimensions (below a certain threshold) while ensuring that the overall element count of the tensor remains unchanged. This operation is particularly useful in developing deep learning models where tensor dimensions might need adjustments before passing through layers or operations.

## Class/Function Definition

The `merge_small_dims` function is described as follows:

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| `tensor_shape` | `List[int]` | The shape of the tensor as a list of integers. | N/A |
| `threshold` | `int` | The threshold on the maximum size of each dimension. | N/A |

## Functionality and Usage

`merge_small_dims` takes in the shape of a tensor and merges dimensions with size less than or equal to a specified threshold. This utility does not affect the data within the tensor; instead, it provides a new tensor shape that can be applied to reshape the tensor.

When to use `merge_small_dims`:

- When the tensor has many small dimensions that can be combined without altering the underlying data structure.
- When optimizing memory layout for tensors for computational efficiency.
- To conform to layer or operation constraints that require a specific number of dimensions in PyTorch (or similar libraries).

### Usage Examples

#### Basic Example

```python
from typing import List
from zeta.ops import merge_small_dims

# Original tensor shape
orig_shape = [2, 3, 1, 5, 1]
# Threshold for maximum size of each dimension after the merge
threshold = 10

# Merging small dimensions
new_shape = merge_small_dims(orig_shape, threshold)
print(new_shape)  # Output: [6, 5]
```

In the example above, the original shape of `[2, 3, 1, 5, 1]` contains small dimensions that can be merged without exceeding the threshold of `10`. The resulting `new_shape` after calling `merge_small_dims` is `[6, 5]`.

#### PyTorch Integration Example

```python
import torch
from zeta.ops import merge_small_dims

# Define a tensor with a shape that includes small dimensions
tensor = torch.rand(2, 3, 1, 5, 1)

# Define the threshold
threshold = 10

# Obtain the new shape
new_shape = merge_small_dims(tensor.size(), threshold)

# Reshape the tensor accordingly
reshaped_tensor = tensor.view(new_shape)

print(reshaped_tensor.size())  # Output: torch.Size([6, 5])
```

In this example, we use PyTorch to define a random tensor with a shape that includes small dimensions. We then obtain a new shape from the `merge_small_dims` function and apply it to the tensor using `.view(new_shape)` method provided by PyTorch.

#### Preventing Dimension Merge Example

```python
from zeta.ops import merge_small_dims

# Original shape that includes a dimension larger than the threshold which should not be merged
orig_shape = [2, 10, 1, 5, 1]
# Threshold for maximum size of each dimension after merge
threshold = 9  # Lower than the size of the second dimension

# Merging small dimensions
new_shape = merge_small_dims(orig_shape, threshold)
print(new_shape)  # Output: [2, 10, 5]
```

Here, the second dimension of size `10` is not merged with any other dimension because it exceeds the threshold of `9`. Only the third, fourth, and fifth dimensions are merged because their combined size (`1 * 5 * 1`) is within the limit.

## Additional Information and Tips

- The function assumes the input shape is valid and does not include validation for negative sizes or non-integer values.
- The first dimension is never merged with any other dimension. This is typically due to the first dimension representing the batch size in most deep learning frameworks.
- The thresholds should be chosen carefully with an understanding of how it may affect subsequent operations that rely on tensor shapes.
- It's recommended to thoroughly verify the new tensor shape with respect to the needs of your specific model or computation graph.

