# multi_dim_cat

The `zeta.ops` library provides a set of operations to manipulate tensor objects flexibly and efficiently. One of the fundamental utilities within this library is the `multi_dim_cat` function. This function serves the purpose of concatenating a list of tensor objects across multiple dimensions, allowing the user to combine tensor splits back into a singular tensor. This operation is particularly useful in scenarios where tensor operations have been parallelized or distributed across multiple processing units and need to be recombined.

## Installation

Before using `zeta.ops`, ensure you have PyTorch installed in your environment.

```bash
pip install torch
```

Once PyTorch is installed, you can include `zeta.ops` functions directly in your project.

## Importing

```python
import torch

from zeta.ops import (  # Assuming zeta.ops is correctly installed and accessible
    multi_dim_cat,
)
```

## Structure & Architecture

The `multi_dim_cat` function aligns with PyTorch's design philosophy, enabling seamless tensor operations with high performance in mind.

### multi_dim_cat

#### Purpose

The `multi_dim_cat` function is designed to merge a list of tensors (split_tensors) across the specified dimensions as indicated by the number of splits for each dimension (num_splits).

#### Parameters

| Parameter     | Type          | Description                             |
| ------------- | ------------- | --------------------------------------- |
| `split_tensors` | `List[Tensor]` | List of tensor splits to be concatenated. |
| `num_splits`    | `List[int]`    | The number of tensor blocks in each corresponding dimension. |

#### Returns

| Return        | Type        | Description  |
| ------------- | ----------- | ------------ |
| `merged_tensor` | `Tensor`    | The tensor resulting from concatenating the input tensor list across the specified dimensions. |

#### Method

```python
def multi_dim_cat(split_tensors: List[Tensor], num_splits: List[int]) -> Tensor:
    # The code implementation is detailed in the source.
```

## Usage Examples

Below are three usage examples that showcase how to use the `multi_dim_cat` function. Each example provides a different scenario to help learners understand how to apply this operation in various contexts.

### Example 1: Basic Concatenation

This example demonstrates a basic usage of `multi_dim_cat` where tensors are concatenated along one dimension.

```python
import torch

from zeta.ops import multi_dim_cat

# Assume we have a list of 3 tensors we wish to concatenate along the 1st dimension
tensor_splits = [torch.randn(2, 3) for _ in range(3)]
num_splits = [3]

# Concatenate tensors
merged_tensor = multi_dim_cat(tensor_splits, num_splits)
print(merged_tensor.shape)  # Expected output: torch.Size([2, 9])
```

### Example 2: Concatenating Across Multiple Dimensions

This example shows how one might concatenate tensor slices across two dimensions.

```python
import torch

from zeta.ops import multi_dim_cat

# Creating a list of 4 tensors with 2 splits across each of two dimensions
tensor_splits = [torch.randn(2, 2) for _ in range(4)]
num_splits = [2, 2]

# Concatenate tensors across two dimensions
merged_tensor = multi_dim_cat(tensor_splits, num_splits)
print(merged_tensor.shape)  # Expected output: torch.Size([4, 4])
```

### Example 3: Reassembling a 3D Tensor from Splits

This example illustrates concatenating splits to reassemble a higher-dimensional tensor from its blocks.

```python
import torch

from zeta.ops import multi_dim_cat

# Imagine we have split a 3D tensor into 8 blocks (2 x 2 x 2)
tensor_splits = [torch.randn(1, 1, 1) for _ in range(8)]
num_splits = [2, 2, 2]

# Concatenate slices to form the original 3D tensor
merged_tensor = multi_dim_cat(tensor_splits, num_splits)
print(merged_tensor.shape)  # Expected output: torch.Size([2, 2, 2])
```

## Tips and Tricks

1. Verify split sizes: Ensure that the number of splits correctly partitions the list of `split_tensors`.
2. Memory considerations: The concatenation of large tensors can be memory-intensive. Plan and structure your tensor operations accordingly.
3. Testing edge cases: Test with various shapes and split configurations to ensure robust behavior of your application when using `multi_dim_cat`.

## Troubleshooting

- If you encounter an assertion error, verify that the number of tensors in `split_tensors` matches the product of `num_splits`.
- Any mismatches in dimensions during concatenation will raise a runtime error. Ensure that all dimensions, except the concatenating dimension, are equal among tensors.

## Conclusion

The `multi_dim_cat` function in `zeta.ops` is an essential utility for tensor manipulation when working with multi-dimensional data. By understanding and appropriately using this function, you'll be empowered to write more efficient and flexible PyTorch code for your complex data processing tasks.

---