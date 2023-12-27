# track_cuda_memory_usage

# Zeta Utils Documentation

The zeta.utils package is designed to simplify and enhance numerous coding tasks related to PyTorch deep learning systems. By using decorators, the package creates a higher order function that wraps standard functions to provide additional capabilities.

This documentation will provide in-depth focus on the `track_cuda_memory_usage` function decorator included in the package. The intent of this documentation is to thoroughly acquaint the user with the usage and function of `track_cuda_memory_usage`.

## Function Definition

The `track_cuda_memory_usage` function is a decorator that, when applied to another function, tracks and logs the CUDA memory usage during the execution of that function. The primary purpose of `track_cuda_memory_usage` is to allow users to understand the GPU memory allocation and usage when executing a given function - a valuable tool for optimizing deep learning models and operations.

This function is especially beneficial when working with large models or data as it allows for efficient memory allocation and monitoring. Using the insights gleaned from this function, users can adjust either their model or their data processing methods to ensure memory efficiency.

```python
def track_cuda_memory_usage(func):
    """
    Name: track_cuda_memory_usage

    Documentation:
    Track CUDA memory usage of a function.

    Args:
    func (function): The function to be tracked.

    Returns:
    function: The wrapped function.
    """
```

## Arguments

|  Argument   |   Data Type   |   Default Value   |   Description   |
|-------------|---------------|-------------------|-----------------|
|    func    |     function    |        N/A          |   The function to be tracked.    |

## Usage examples

```python
from zeta.utils import track_cuda_memory_usage
import torch

# Define the function that you wish to track
@track_cuda_memory_usage
def create_empty_tensor(size):
    return torch.empty(size=(size, size)).cuda()

create_empty_tensor(1000)
```

In this example, the decorator `@track_cuda_memory_usage` is used to track the CUDA memory usage during the execution of the function `create_empty_tensor`, which creates an empty tensor on the GPU. On execution of this function, CUDA memory usage details will be logged.

Here's an example tracking the memory usage while training a model, which could help in understanding and improving the efficiency of a training loop.

```python
from zeta.utils import track_cuda_memory_usage
import torch
from torchvision.models import resnet18
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

model = resnet18().cuda()

optimizer = SGD(model.parameters(), lr=0.01)

# Define a simple train loop
@track_cuda_memory_usage
def simple_train_loop(dataloader, model, optimizer):
    loss_function = CrossEntropyLoss()
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

simple_train_loop(your_dataloader, model, optimizer)
```

In this example, we define a simple training loop for a model and use the `@track_cuda_memory_usage` decorator to monitor the CUDA memory usage for each iteration of the loop.

## Additional Usage Tips

Prior to running any operation, the function forces PyTorch to wait for all currently pending CUDA operations to finish with `torch.cuda.synchronize()`. This ensures that all previously allocated memory is factored into the calculation before the execution of `func`.

It's crucial to note that GPU memory usage is often non-deterministic due to factors such as CUDA's memory management mechanisms as well as multi-threaded operations.

## Conclusion

Understanding how `track_cuda_memory_usage` works can make a significant difference in optimizing and diagnosing memory-related issues in a PyTorch project. This utility is paramount to developers who work with large data and models. It's a handy tool that makes memory debugging and tracking accessible and manageable.
