# cast_if_src_dtype

# Module Name: `cast_if_src_dtype`
****
# Description
`cast_if_src_dtype` is a utility function that checks the data type (`dtype`) of a given tensor. If the tensor's `dtype` matches the provided source `dtype` (`src_dtype`), the function will cast the tensor to the target `dtype` (`tgt_dtype`). After the casting operation, the function returns the updated tensor and a `boolean` flag indicating whether the tensor data type was updated.

This function provides a convenient way to enforce specific data types for torch tensors.

# Class/Function Signature in Pytorch

```python
def cast_if_src_dtype(
    tensor: torch.Tensor, src_dtype: torch.dtype, tgt_dtype: torch.dtype
):
    updated = False
    if tensor.dtype == src_dtype:
        tensor = tensor.to(dtype=tgt_dtype)
        updated = True
    return tensor, updated
```
# Parameters

| Parameter | Type | Description |
| :-------- | :--: | :---------- |
| `tensor`  | `torch.Tensor` | The tensor whose data type is to be checked and potentially updated. |
| `src_dtype` | `torch.dtype` | The source data type that should trigger the casting operation. |
| `tgt_dtype` | `torch.dtype` | The target data type that the `tensor` will be cast into if the source data type matches its data type. |

# Functionality and Use
**Functionality:** `cast_if_src_dtype` takes in three parameters: a tensor, a source data type, and a target data type. If the data type of the tensor equals the source data type, the function casts this tensor to the target data type. The function then returns both the potentially modified tensor and a flag indicating whether the cast was performed.

**Usage**: This utility function is used when certain operations or functions require inputs of a specific data type. A common scenario is when tensors with floating-point data types need to be converted to integers or vice versa.

# Usage Examples
Below are some examples of how the function could be used:

## Example 1
```python
import torch

from zeta.utils import cast_if_src_dtype

# Given: a float tensor
tensor = torch.tensor([1.0, 2.0, 3.0])

# We want to convert it to integer type tensor if its data type is float32
tensor, updated = cast_if_src_dtype(tensor, torch.float32, torch.int32)

print(tensor)  # tensor([1, 2, 3], dtype=torch.int32)
print(updated)  # True
```

## Example 2
```python
import torch

from zeta.utils import cast_if_src_dtype

# Given: an integer tensor
tensor = torch.tensor([1, 2, 3])

# We want to convert it to float type tensor if its data type is int32
tensor, updated = cast_if_src_dtype(tensor, torch.int32, torch.float32)

print(tensor)  # tensor([1.0, 2.0, 3.0])
print(updated)  # True
```

## Example 3
```python
import torch

from zeta.utils import cast_if_src_dtype

# Given: an integer tensor
tensor = torch.tensor([1, 2, 3])

# If the data type is not equal to the source data type, the tensor will remain the same
tensor, updated = cast_if_src_dtype(tensor, torch.float32, torch.int32)

print(tensor)  # tensor([1, 2, 3])
print(updated)  # False
```
# Resources and References
For more information on tensor operations and data types in PyTorch, refer to the official PyTorch documentation:

- [PyTorch Tensor Operations](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch Data Types](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype)

# Note
The `cast_if_src_dtype` function doesn't modify the original tensor in-place. Instead, it creates a new tensor with the updated data type. Keep that in mind during function calls, and be sure to substitute the original tensor with the returned tensor to reflect the change in the rest of your code.
