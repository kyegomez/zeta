# Module Name: PScan

## Overview and Introduction

The PScan class is an implementation of the parallel scan operation in PyTorch. The code is based on Francois Fleuret’s pscan but has been written in an iterative way rather than recursively. The backward pass has been rewritten to improve efficiency, and the code provides a more detailed and efficient implementation of the parallel scan operation in PyTorch.

This documentation will provide a comprehensive overview of the PScan class, including details about its purpose, class definition, functionality, usage examples, and additional information for utilizing the functionality provided by the class.

## Class Definition

The PScan class is implemented as a torch.autograd.Function, which allows it to be directly used as an operation within PyTorch. The key parameters of the class include A_in and X_in, which represent input tensors, and H, which represents the resulting output of the parallel scan operation. The class also includes methods for both the forward and backward passes, using them to compute the outputs and gradients of the operation.


## Functionality and Usage

The parallel scan operation is applied using the forward method of the PScan class. The parallel scan takes two input tensors A_in and X_in and performs a parallel scan operation on them to produce the output tensor H. Additionally, the backward method is used to calculate the gradients of the output with respect to the inputs, which are returned as gradA and gradX.

The parallel scan operation uses an iterative approach to efficiently compute the parallel scan of the input tensors, reducing the time complexity compared to a recursive implementation. The forward and backward passes ensure that the output and gradients of the operation are correctly calculated, making it suitable for differentiable optimization procedures.

### Code Snippet for Usage
```python
import torch
from zeta.nn import PScan

# Create input tensors
x = torch.randn(2, 3, 4, 5, requires_grad=True)
y = torch.randn(2, 3, 4, 5, requires_grad=True)

# Apply the parallel scan operation
model = PScan.apply(x, y)

# Perform backpropagation to compute gradients
model.sum().backward()
print(x.grad)
print(y.grad)
```

## Additional Information and Tips

- The PScan class is based on the Blelloch version of the parallel scan operation.
- The code is written for efficient and differentiable parallel scan computations in PyTorch.
- It is important to clone input tensors before using the PScan operation.

## References and Resources

- For a detailed explanation with examples, see the pscan.ipynb document included in the repository.
- For further details about PyTorch and differentiable programming, refer to the official PyTorch documentation.

This comprehensive documentation provides a detailed overview of the PScan class, including its implementation, purpose, functionality, usage, and additional tips. The class serves as a valuable tool for efficiently computing parallel scans in PyTorch and is aimed at users who seek to utilize differentiable operations within the PyTorch framework.
