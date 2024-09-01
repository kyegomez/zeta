# QuickGELUActivation
## Overview

The QuickGELUActivation class is a part of the Neural Network(NN) module that applies a Gaussian Error Linear Unit (GELU) approximation. GELU can be viewed as a smoother version of the popular activation function, ReLU. The approximate version of GELU used in this class is fast although somewhat less accurate than the standard GELU activation.

The GELU activation function can be used as an alternative to other popular activation functions like ReLU and Sigmoid while training deep learning models. The importance of GELU in the context of deep learning comes from its unique properties which includes non-monotonicity that allows for complex transformations.

## Class Definition

The QuickGELUActivation class is defined as shown below:

```python
class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
```

The class extends the Module class from the pyTorch library. It does not take any input parameters during initialization.

## Method Definitions

The class has a single method named forward.

### forward

This function is responsible for applying the GELU approximation to the input tensor.

```python
def forward(self, input: Tensor) -> Tensor:
    return input * torch.sigmoid(1.702 * input)
```

**Parameters:**

| Name | Type |Description |
| --- | --- | --- |
| **input** | Tensor | The input tensor to which the GELU approximation will be applied. |

**Return Type:** Tensor 

**Returns:** The output tensor after applying the GELU approximation.

## Meta-information

The function uses a torch inbuilt function *sigmoid* to apply the GELU approximation. The parameter 1.702 in the sigmoid function is chosen as it approximates the GELU function very closely. It should be noted that this approximation may not be exactly equal to the standard GELU and hence, could be somewhat inaccurate.

## Example Code

Below is a simple example showing how to use QuickGELUActivation to apply a GELU approximation to a tensor input:

```python
import torch
from torch import nn

from zeta.nn import QuickGELUActivation

# create an instance of QuickGELUActivation
activation = QuickGELUActivation()

# create a tensor
x = torch.rand(3)

# apply GELU activation
output = activation(x)

print(output)
```

In this code, we first create a tensor using the `rand` method from pyTorch. Next, an instance of the QuickGELUActivation class is created and the GELU approximation is applied to the tensor.

Further, it is advised to use this GELU activation function in the scenario where quick approximation is more advantageous than a slightly more accurate result. It can be used with any model architecture where an activation function is needed. It may provide better results in certain scenarios compared to typical activation functions like ReLU. 

For more details, you can refer to the [GELU activation paper](https://arxiv.org/abs/1606.08415) and the [approximation method](https://github.com/hendrycks/GELUs). 

This class is not a direct replacement for the torch.nn.GELU and should be used considering the trade-off between speed and accuracy. Please also refer to the official [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html) documentation for more information on activation functions in PyTorch.
