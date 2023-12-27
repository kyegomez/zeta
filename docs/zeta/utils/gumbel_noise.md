# gumbel_noise

# Module Name: Gumbel Noise

Function Name: gumbel_noise(t)

```python
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))
```
This function generates Gumbel noise, a type of statistical noise named after the Emil Julius Gumbel who was a German statistician, applied to a tensor 't' with similar attributes. It generates a tensor with the same size as 't', filled with random numbers uniformlly distributed between 0 (inclusive) and 1 (exclusive). Then, the Gumbel noise is computed which is a perturbation method to draw samples from discrete distributions.

The Gumbel distribution is used in sampling methods, for example in the Gumbel-Softmax trick, for producing one-hot encodings or to sample from a discrete distribution with an unspecified number of classes.

Parameters:
- t (torch.Tensor) : Input tensor.

Return:
- Tensor: Gumbel noise added tensor with the same type as t. The equals to negative logarithm of negative logarithm of uniform noise.

## Example:

```python
import torch
from math import log

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# Creating a tensor
x = torch.tensor([2.0, 1.0, 3.0, 4.0])
print("Original Tensor: ",x)

# Applying gumbel noise
y = gumbel_noise(x)
print("Tensor after applying Gumbel noise function: ",y)
```
## Issues and Recommendations

- It should be noted that the function torch.zeros_like() can be replaced by the torch.empty_like() function if wanting to save time when generating the tensor. The former sets all values as zeros while the latter does not initialize the values, a step that isn't necessary since we are just overwriting these values with uniform noise.

- Note that the function is computing the logarithm of noise. In the case where noise is very low and close to zero, the inner logarithm will give negative infinity. Subsequently, negative of negative infinity is positive infinity. Users should be aware of potential overflow issues in their computations.
   
- If the function is used in machine learning models for training, it should be noted that the function is not different
