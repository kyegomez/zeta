# log

# zeta.utils.log

## Introduction

The `log` function serves as a small utility helper to calculate the natural logarithm of a tensor using PyTorch's `torch.log` function, while safeguarding against division by zero error by setting a minimum clamp value.

The minimum clamp value serves as a protection from taking the log of 0 which would result in undefined mathematical operation (division by zero). The aim of this is to ensure computational stability, especially in context where the input tensor contains zero or near-zero values. 

## Function Definition

This function, `zeta.utils.log(t, eps=1e-20)`, has the following parameters:

* `t` : A PyTorch tensor that the logarithm will be taken from. This tensor can have any shape.
* `eps` (default: `1e-20`): A small value which sets the minimum value for clamping. This essentially serves as a "safety net" preventing the input tensor from being zero or negative, which would result in an error when we take the log.

## Return Value
The function `zeta.utils.log(t, eps=1e-20)` returns a tensor of the same shape, where each element represents the natural logarithm of the corresponding element from the input tensor `t` with a minimum clamp established by `eps`.

## Functionality and Usage

The implementation of the function is as follows:

```python
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))
```

`t.clamp(min=eps)` restricts the values within tensor `t` to be greater or equal to the `eps` value. This is to avoid any fraudulent computations involving negative or zero values when the logarithm function is applied to these clamp restricted values by `torch.log`.

This function is typically used in situations where it's necessary to calculate the natural log of tensor values in machine learning models, especially in those contexts where the input tensor might contain zero or near-zero values due to computations in the model or the nature of the input data.

Here is a simple example usage of `zeta.utils.log`:

```python
import torch

import zeta.utils as zutils

t = torch.tensor([0.0, 0.1, 1.0, 10.0])
res = zutils.log(t)

print(res)
```
```console
tensor([-46.0517,  -2.3026,   0.0000,   2.3026])
```

**Note**: As seen in the example above, instead of `inf` which is typically what we get by applying log to zero, our log utility function gives a large negative number (-46.0517), thanks to the `eps` clamping.

## Additional Tips

As mentioned earlier, the purpose of the `eps` parameter is to prevent possible mathematical errors when taking the log of zero or negative numbers. However, the default value of `eps` is set to `1e-20` which can be too small in some contexts, leading to extreme values when taking the log.

Depending on the scale and the nature of your data, it may be useful to adjust `eps` to a larger value to avoid very large negative numbers but remember, setting `eps` too high might introduce a bias. As always, it’s a balance and the right value of `eps` depends on your specific situation.

Here is another example of how adjusting `eps` can affect your results:

```python
import torch

import zeta.utils as zutils

t = torch.tensor([0.0, 0.1, 1.0, 10.0])
res = zutils.log(t, eps=1e-10)

print(res)
```
```console
tensor([-23.0259,  -2.3026,   0.0000,   2.3026])
```

In this example, by setting `eps` to `1e-10` we've effectively "softened" the result from applying log to zero from `-46.0517` to `-23.0259`.
