# log

# Module Name: zeta.utils.log

## Table of Contents

- [Introduction](#Introduction)
- [Arguments](#Arguments)
- [Methods](#Methods)
- [Examples](#Examples)
- [Tips](#Tips)
- [References](#References)

## Introduction
This document is a detailed and comprehensive guide on how to use the `log` module that exists within the `zeta.utils` library.

`log` is a utility function signature within the `zeta.utils` library, which specifically takes in a PyTorch Tensor and returns its natural logarithm (base `e`) after applying a clamp operation. Clamping refers to setting the value within an interval `min` and `max`. Here we only want to ensure that the tensor values are not lower than a small value `eps` which is often taken to prevent division by zero or log of zero errors.

## Arguments

This function accepts two arguments: `t` and `eps`.

| Argument | Type | Default | Description |
| -------  | ---- | ------- | ----------- |
| `t` | torch.Tensor  | N/A | The input tensor on which the natural logarithm operation is performed. |
| `eps` | float | 1e-20 | A very small value to which tensor values are set if they are less than `eps`. This helps in avoiding computation errors when we evaluate log of these tensor values.| 

All arguments are compulsory, but you can omit `eps` during a function call; in this case, its default value (1e-20) would be used.

## Methods

`log` is a standalone function and does not have any class or instance-specific methods. 

To call it, use `zeta.utils.log(t, eps)` where `t` is the tensor and `eps` is the optional small value as explained above. 

## Examples

These examples demonstrate how to utilize the `log` function within the `zeta.utils` library.

- First, import the necessary libraries:

```python
    import torch
    from zeta.utils import log
```

- Using `log` function with a simple tensor:

```python
    # Define tensor
    t = torch.tensor([0.0, 1.0, 2.0, 3.0])
    
    # Apply log transformation
    log_t = log(t)  

    print(log_t)  
```
The expected output should
