# cast_if_src_dtype

# Zeta Utils Documentation

## Table of Contents

1. [cast_if_src_dtype](#cast_if_src_dtype)

<a name='cast_if_src_dtype'></a>
## cast_if_src_dtype 
`cast_if_src_dtype(tensor, src_dtype, tgt_dtype)`

This function is utilized to change the data type (`dtype`) of a given tensor if the current data type matches the source data type specified. The process of changing one type to another is called "Casting" in both general computing and PyTorch. 

The function requires three arguments: `tensor`, `src_dtype`, and `tgt_dtype`.

You would want to use this function when working with different data types in PyTorch. For instance, it ensures uniform data types across tensors for operations that require tensors of the same type. With this utility function, we can cast our tensor to the desired type only if the source type matches our tensor.

Below is the table summary of the arguments of this function:

| Argument | Type | Description |
| :- | :- | :- |
| tensor   | torch.Tensor | The input tensor whose data type may need to be changed. |
| src_dtype | torch.dtype | The source data type to be matched. If the current data type of the tensor matches this, it will be changed. |
| tgt_dtype | torch.dtype | The target data type to which the tensor will be casted if its current data type matches the source data type. |

The function returns two variables:

 1. The potentially updated tensor.
 2. A boolean variable (`True` if the tensor was updated, `False` if not).

### Examples

#### Basic Example

Here's an example of how it works. We'll start by importing the necessary tools:

```python
import torch
from zeta.utils import cast_if_src_dtype
```
Now, let's say we're given the following tensor of integers:

```python
t1 = torch.tensor([1, 2, 3, 4, 5])
print(t1.dtype)  # Outputs torch.int64
```
We want to cast this tensor to `float32` only if it's current dtype is `int64`. Here's how to do it:

```python
t1, updated = cast_if_src_dtype(t1, torch.int64, torch.float32)

print(t1.dtype)  # Outputs torch.float32
print(updated)  # Outputs True
```
In this
