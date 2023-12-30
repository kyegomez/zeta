# multi_dim_split

The `multi_dim_split` function is a utility designed to chunk a given tensor across multiple dimensions based on specified split sizes. This operation is particularly useful in scenarios where one needs to divide a tensor into smaller, more manageable blocks for parallel processing or specific algorithmic purposes.

Understanding how to split tensors appropriately is crucial in machine learning and scientific computing tasks. Efficient data manipulation can significantly impact the performance and scalability of models and algorithms.

## Overview
The `multi_dim_split` function works by accepting a tensor and a list of sizes that determine how the tensor should be divided along each dimension. It sequentially applies the splitting operation for each dimension specified by the splits. The function ensures that the tensor is divided into blocks, each with the specified size along the corresponding dimension.

## Function Definition

```python
def multi_dim_split(
    tensor: torch.Tensor,
    splits: List[int],
) -> List[torch.Tensor]:
```

### Parameters:

| Parameter | Type             | Description                                                                                           |
|-----------|------------------|-------------------------------------------------------------------------------------------------------|
| tensor    | `torch.Tensor`   | The input tensor to be split.                                                                         |
| splits    | `List[int]`      | A list of sizes for each block or chunk along each dimension.                                         |

### Returns:

| Return Value   | Type                 | Description                                                                    |
|----------------|----------------------|--------------------------------------------------------------------------------|
| split_tensors  | `List[torch.Tensor]` | A list of tensors resulting from splitting the input tensor along dimensions.   |

## Usage and Examples

### Example 1: Basic Splitting
```python
import torch
from typing import List
from zeta.ops import multi_dim_split

# Create a simple 3D tensor
tensor_3d = torch.randn(4, 6, 8)

# We want to split the tensor into blocks of sizes 2x3x4
splits = [2, 3, 4]

# Perform the split operation
split_tensors = multi_dim_split(tensor_3d, splits)

# Output the shape of each split tensor
for i, split_tensor in enumerate(split_tensors):
    print(f"Block {i+1}: {split_tensor.size()}")
```

### Example 2: Splitting Along Specific Dimensions
```python
import torch
from typing import List
from zeta.ops import multi_dim_split

# Create a 2D tensor
tensor_2d = torch.randn(10, 12)

# Split the tensor into blocks of 5 along the first dimension only
splits = [5]

# Perform the split operation
split_tensors = multi_dim_split(tensor_2d, splits)

# View the result
for i, split_tensor in enumerate(split_tensors):
    print(f"Split {i+1}: {split_tensor.size()}")
```

### Example 3: Splitting a High-Dimensional Tensor
```python
import torch
from typing import List
from zeta.ops import multi_dim_split

# Create a 4D tensor
tensor_4d = torch.randn(8, 12, 16, 20)

# Split the tensor into 2x3x4x5 blocks
splits = [2, 3, 4, 5]

# Perform the split
split_tensors = multi_dim_split(tensor_4d, splits)

# Display the shapes of the resulting tensors
for i, split_tensor in enumerate(split_tensors):
    print(f"Chunk {i+1}: {split_tensor.size()}")
```

## Functionality and Architecture

The `multi_dim_split` function's architecture involves iterative splitting of the input tensor along specified dimensions. The initial input is a single tensor that is processed in a loop, where each iteration handles splitting along one dimension, creating intermediate lists of tensors.

First, a list containing the original tensor is created. This ensures that the subsequent loop can iterate over either the original tensor or the tensors resulting from previous splits. Then the function loops over the dimensions corresponding to the provided `splits` list. Each iteration applies `torch.split` to every tensor in the list across the current dimension.

The `torch.split` operation divides a tensor into chunks along a specified dimension, here defined by the `split` sizes. The resulting split tensors are then collected into a new list, replacing the original list. This process continues until all dimensions have been handled, resulting in a final list of split tensors.

This architecture allows `multi_dim_split` to be flexible and handle tensors of any shape, provided the `splits` argument correctly corresponds to the tensor's dimensions.

## Additional Information and Tips

- Ensure that the sum of the sizes specified in `splits` for each dimension does not exceed the size of the tensor in that dimension. Otherwise, you may encounter errors or unexpected behavior.
- If an exact split is not possible because the dimension size is not divisible by the split size, `torch.split` will produce a smaller last block for that dimension.
- The order of the sizes in the `splits` list should match the dimensions of the tensor you wish to split. That is, the first number in `splits` applies to dimension 0 of the tensor, the second number to dimension 1, and so on.
- The function uses a list comprehension to flatten the list of split tensors after each dimension is processed. Understanding list comprehensions and their performance implications is valuable when working with these types of operations.

## Conclusion and References

The `multi_dim_split` function is a powerful tool for tensor manipulation, allowing users to split tensors into smaller blocks across multiple dimensions efficiently. By understanding its parameters and functionality, developers can employ this function in a variety of data manipulation and parallel computing tasks.

For more information on the underlying `torch.split` function and tensor operations in PyTorch, refer to the official PyTorch documentation:

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- torch.split: https://pytorch.org/docs/stable/generated/torch.split.html

Understanding the `multi_dim_split` function provides deeper insights into efficient data processing, paving the way for more advanced tensor operations and algorithm implementations.