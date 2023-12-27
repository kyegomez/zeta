# print_cuda_memory_usage

# `zeta.utils`: print_cuda_memory_usage

# Purpose and Functionality

This is a Python context manager function designed for tracking and reporting CUDA (Compute Unified Device Architecture) memory usage during GPU-accelerated operations in PyTorch. CUDA is a parallel computing platform and application programming interface (API) model created by NVIDIA which allows software developers to use a CUDA-enabled graphics processing unit (GPU) for general-purpose processing.

`print_cuda_memory_usage` monitors the GPU memory consumption before and after the context block of code that it wraps. Upon exit of the context block, it calculates the change in memory usage and outputs it in gigabytes.

# Function Definition

```python
from contextlib import contextmanager
import torch

@contextmanager
def print_cuda_memory_usage():
    initial_memory = torch.cuda.memory_allocated()
    try:
        yield
    finally:
        memory_usage = torch.cuda.memory_allocated() - initial_memory
        memory_usage_gb = memory_usage / (1024**3)
        print(f"CUDA memory usage: {memory_usage_gb:.2f} GB")
```

The `@contextmanager` decorator transforms `print_cuda_memory_usage` into a factory function that returns a context manager. When entering the context block, it records the starting GPU memory usage. It then yields control to the contents of the context block. Upon exiting the block, it records the final GPU memory usage, calculates the difference, and prints it to the standard output.

# Arguments

`print_cuda_memory_usage` doesn't take any arguments.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| None     | None | None        |

# Usage

Here are some examples on how `print_cuda_memory_usage` can be used:

## Example 1: Basic Usage

```python
x = torch.randn((10000, 10000), device='cuda')

with print_cuda_memory_usage():
    y = x @ x.t()  # Large matrix multiplication
```

In this example, a large tensor `x` is allocated on the GPU, and then a large matrix multiplication is performed inside the `print_cuda_memory_usage` context. The increase in GPU memory usage resulting from this operation will be printed.

## Example 2: Exception Handling

```python
x = torch.randn((10000, 10000), device='cuda')

try:
    with print_cuda_memory_usage():
        y = x @ x.t()  # Large matrix multiplication
        raise Exception("Some Exception")
except Exception as e:
    print(f"Caught an exception: {e}")
```

In this example, an exception is raised inside the `print_cuda_memory_usage` context. Regardless of the exception, `print_cuda_memory_usage` will still correctly compute and print the CUDA memory usage before the exception is propagated.

## Example 3: Nesting Usage

```python
x = torch.randn((10000, 10000), device='cuda')

with print_cuda_memory_usage():
    y = x @ x.t()  # Large matrix multiplication
    with print_cuda_memory_usage():
        z = y @ y.t()  # Even larger matrix multiplication
```

In this example, `print_cuda_memory_usage` contexts are nested, allowing you to separately track the GPU memory usage of different parts of your code.

# Notes

The `print_cuda_memory_usage` function requires PyTorch to be run with CUDA enabled and a CUDA-enabled GPU to be available. If either of these conditions are not met, `torch.cuda.memory_allocated()` will raise a `RuntimeError` and the function will not work as intended.

Also, `print_cuda_memory_usage` only tracks the GPU memory that is allocated and managed by PyTorch, it doesn't account for any memory directly allocated by CUDA via methods outside of PyTorch's control.

Finally, `print_cuda_memory_usage` gives an indication of the additional memory used by a specific block of code. However, the exact details of memory management on the GPU can be complex, depending on multiple factors such as how PyTorch allocates and caches memory, the specific GPU hardware, the CUDA version, and other aspects of the system configuration. It also does not account for the memory used by non-PyTorch CUDA libraries or other processes sharing the same GPU.
