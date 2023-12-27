# print_cuda_memory_usage

# Module Name: zeta.utils

The `zeta.utils` module hosts a utility function `print_cuda_memory_usage()`, a Python context manager function to print the amount of CUDA memory that a specific block of code uses. This function is particularly useful in deep learning applications, where memory management is crucial due to the high usage of memory by models and datasets.

The `print_cuda_memory_usage()` function uses PyTorch to perform memory operations, one of the popular open-source deep learning platforms, and it requires an NVIDIA GPU and CUDA toolkit already installed, because CUDA operations require access to a CUDA-enabled GPU.

# Function Definition: print_cuda_memory_usage()

## Function Signature
```python
@contextmanager
def print_cuda_memory_usage():
```

## Function Description

This function is a context manager function that prints the CUDA memory usage of the code block that calls this function. The memory usage is calculated by subtracting the amount of CUDA memory allocated at the end of the code block from the amount of CUDA memory allocated immediately before executing the code block. The resultant memory usage is then converted from bytes to gigabytes and printed to the console.

## Function Parameters and Return Values

Since `print_cuda_memory_usage()` is a context manager function, it does not take parameters nor return any values. It is intended to be used with the `with` statement in Python.

| Parameter Name | Type | Description | Default Value |
|:--------------:|:----:|:-----------:|:-------------:|
| - | - | - | - |

| Return Name | Type | Description |
|:-----------:|:----:|:------------:|
| - | - | - |

## Example Code

The following are example codes that show how to use the function:

### Example: Memory usage of a small tensor 

We first import the necessary libraries:

```python
import torch
from zeta.utils import print_cuda_memory_usage
```

Next, we use the `print_cuda_memory_usage()` function to get the CUDA memory usage of creating a small tensor with PyTorch.

```python
with print_cuda_memory_usage():
    a = torch.tensor([1.]).cuda()
```

### Example: Memory usage of a large tensor

In this example, we again use the `print_cuda_memory_usage()` function to observe the CUDA memory usage but with a larger tensor with PyTorch.

```python
with print_cuda_memory_usage():
    a = torch.rand(1024
