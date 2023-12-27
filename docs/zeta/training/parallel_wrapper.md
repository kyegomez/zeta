# `ParallelWrapper`
===============

The `ParallelWrapper` class is a simple wrapper designed to facilitate the use of data parallelism in PyTorch. It is particularly suited for transformer architectures. The class wraps a given neural network model and allows it to be moved to a specified device. If data parallelism is enabled, the model is wrapped in PyTorch's `nn.DataParallel` class, which splits the input across the specified device's GPUs and parallelizes the forward pass.

## Class Definition
----------------

```
class ParallelWrapper:
    def __init__(self, model: nn.Module, device: str = "cuda", use_data_parallel: bool = True):
        pass

    def forward(self, *args, **kwargs):
        pass

    def to(self, device: str):
        pass

    def __getattr__(self, name: str):
        pass
```

## Parameters
----------

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `nn.Module` | - | The neural network model to be parallelized. |
| `device` | `str` | `"cuda"` | The device to which the model should be moved. |
| `use_data_parallel` | `bool` | `True` | A flag to indicate whether to use data parallelism or not. |

## Methods
-------

### `__init__(self, model, device="cuda", use_data_parallel=True)`

The constructor for the `ParallelWrapper` class. It initializes the instance and moves the model to the specified device. If data parallelism is enabled and more than one GPU is available, it wraps the model in `nn.DataParallel`.

### `forward(self, *args, **kwargs)`

The forward method for the `ParallelWrapper` class. It simply calls the forward method of the wrapped model with the provided arguments and keyword arguments.

### `to(self, device)`

This method moves the model to the specified device and updates the `device` attribute of the instance.

### `__getattr__(self, name)`

This method redirects attribute access to the internal model to allow direct access to its methods and properties.

# Usage Examples
--------------

### Example 1: Basic Usage

```python
import torch.nn as nn
from zeta.training import ParallelWrapper  

# Define a model
model = nn.Linear(512, 512)

# Wrap the model
model = ParallelWrapper(model)

# Now you can use the model as usual
input = torch.randn(128, 512)
output = model(input)
```


### Example 2: Using a Different Device

```python
import torch.nn as nn
from zeta.training import ParallelWrapper  

# Define a model
model = nn.Linear(512, 512)

# Wrap the model and move it to CPU
model = ParallelWrapper(model, device="cpu")

# Now you can use the model as usual
input = torch.randn(128, 512)
output = model(input)
```


### Example 3: Disabling Data Parallelism

```python
import torch.nn as nn
from zeta.training import ParallelWrapper  

# Define a model
model = nn.Linear(512, 512)

# Wrap the model and disable data parallelism
model = ParallelWrapper(model, use_data_parallel=False)

# Now you can use the model as usual
input = torch.randn(128, 512)
output = model(input)
```


# Note
----

The `ParallelWrapper` class is a utility that simplifies the use of data parallelism in PyTorch. It does not provide any additional functionality beyond what is already provided by PyTorch's `nn.DataParallel` class. It is intended to be used as a convenience wrapper to reduce boilerplate code.