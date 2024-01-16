## Module/Function Name: Parallel

The `Parallel` class is a module that applies a list of functions in parallel and sums their outputs. This is particularly useful when you need to concurrently apply multiple operations to the same input and aggregate the results.

### Parameters:
The `Parallel` class can take a variable number of functions as input, which will be applied in parallel. The details for each function is provided when they are passed into the `Parallel` constructor, which then forms an `nn.ModuleList` to keep track of them.

### Usage Example:
Below is an example of how to use the `Parallel` class. The example demonstrates creating an instance of `Parallel` with two `nn.Linear` modules and running a randomly generated input through both those linear modules in parallel.

```python
import torch
from torch import nn
from zeta.nn import Parallel

# Define two Linear modules
fn1 = nn.Linear(10, 5)
fn2 = nn.Linear(10, 5)

# Create a Parallel instance
parallel = Parallel(fn1, fn2)

# Generate a random input tensor
input = torch.randn(1, 10)

# Pass the input through the parallel functions and aggregate the results
output = parallel(input)
```

### Overview and Introduction:

The `Parallel` class provides a way to apply a list of functions in parallel and then sum their outputs. It is widely applicable in scenarios where you need to concurrently apply multiple transformations to the same input data.

The purpose of this module is to simplify the process of applying multiple operations to a given input tensor simultaneously and seamlessly aggregating the results. This is achieved by leveraging the `nn.ModuleList` to organize and execute the passed functions in a parallel manner, and then summing the outputs to provide a single combined result.

By using the `Parallel` class, users can avoid repetitive code and streamline the process of applying multiple transformations to their input data, leading to cleaner, more organized code with minimal redundancy and better maintainability.
