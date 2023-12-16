# `track_cuda_memory_usage`

`track_cuda_memory_usage(func)`

A decorator function for tracking CUDA memory usage of a PyTorch function. It measures the amount of CUDA memory allocated before and after the execution of the function, logs the difference, and handles any potential errors during the function execution.

### Parameters:

- `func` (callable): The function to be decorated. This should be a function that performs operations using PyTorch with CUDA support.

### Returns:

- `callable`: The wrapped function, which when called, executes the original function with added CUDA memory tracking and logging.

### Usage:

This decorator can be applied to any function that is expected to run operations using PyTorch with CUDA. To use the decorator, simply place `@track_cuda_memory_usage` above the function definition.

### Example:

```python
@track_cuda_memory_usage
def my_cuda_function(x):
    # Some operations using PyTorch and CUDA
    return x * x

# Example usage
x = torch.randn(1000, 1000, device='cuda')
result = my_cuda_function(x)
```

In this example, `my_cuda_function` is a simple function that squares its input. The decorator logs the amount of CUDA memory used during the function's execution.

### Logging Output:

The decorator logs two types of messages:

1. **Memory Usage Log**: After the function execution, it logs the amount of CUDA memory used by the function. The log is at the INFO level.
   
   Example: `2023-03-15 10:00:00,000 - INFO - CUDA memory usage for my_cuda_function: 4000000 bytes`

2. **Error Log**: If an error occurs during the function execution, it logs the error message at the ERROR level and raises the exception.

   Example: `2023-03-15 10:00:00,000 - ERROR - Error during the execution of the function: RuntimeError(...)`

### Error Handling:

- If CUDA is not available, a warning is logged, and the function runs without memory tracking.
- If an error occurs during the execution of the function, the error is logged, and the exception is re-raised after the memory usage log.

### Notes:

- The decorator uses `torch.cuda.synchronize()` before and after the function execution to ensure accurate measurement of memory usage. This synchronization can introduce some overhead and should be considered when profiling performance-critical code.
- The memory usage reported is the difference in memory allocation on the current CUDA device before and after the function execution. It does not account for memory deallocation that might occur within the function.
