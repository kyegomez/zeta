# track_cuda_memory_usage

# Module/Function Name: track_cuda_memory_usage

This function `track_cuda_memory_usage` is a Python decorator specifically designed to keep track of the GPU memory usage in PyTorch when a different function is called. This provides an easy way of monitoring the CUDA memory usage during the run time of a function, which can help spec out hardware requirements and catch any unusual memory usage patterns indicative of a memory leak.

## Function Definition

```py
def track_cuda_memory_usage(func):
```

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| func | Function | The function whose CUDA memory usage is to be tracked |

### Returns

The function returns a wrapped function. The returned function behaves the same as the passed function (`func`), but it also logs the CUDA memory usage when the function is called.

| Return Value | Type | Description |
| --- | --- | --- |
| Wrapper Function | Function | The wrapped function that behaves the same as the passed function, but also logs the CUDA memory usage |

## Functionality and Usage

The `track_cuda_memory_usage` function wraps the passed function (`func`) and monitors its CUDA memory usage. It does this by checking the GPU memory usage before and after the function runs. If there is an increase in the memory usage, the function logs this change.

This function can be used to debug cases where there are memory leaks in your PyTorch model. It can be especially useful if you're running out of GPU memory but don't know why.

Remember that this is a decorator function and should be used as one. It can be applied to any other function like so:

```python
@track_cuda_memory_usage
def my_func():
    # Function body here
    # This function will now have its CUDA memory usage tracked
    pass
```

## Example of Usage

In the following example, we define a simple PyTorch model and use the `track_cuda_memory_usage` decorator to keep track of the model’s memory usage.

```python
import torch
import torch.nn as nn
import logging

# Creating simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)

# Defining train function
@track_cuda_memory_usage
def train(model, data):
    model.train()

