# ClippedGELUActivation


The ClippedGELUActivation class is designed to clip the possible output range of Gaussian Error Linear Unit (GeLU) activation between a given minimum and maximum value. This is specifically useful for the quantization purpose, as it allows mapping negative values in the GeLU spectrum. To learn more about the underlying concept, you can refer to an academic paper titled [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf).

The original implementation of the GeLU activation function was introduced in the Google BERT repository. Note that OpenAI GPT's GeLU is slightly different and gives slightly different results.

## Class Definition

The ClippedGELUActivation class inherits from the `nn.Module` in PyTorch.

```python
class ClippedGELUActivation(nn.Module):
    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)
```

## Class Arguments

| Argument |   Type  |                                 Description                                  |
|:--------:|:-------:|:----------------------------------------------------------------------------:|
|    min   |  float  |   The lower limit for the output of GeLU activation. It should be less than `max`  |
|    max   |  float  |   The upper limit for the output of GeLU activation. It should be greater than `min` |

Note: If `min` is greater than `max`, a ValueError will be raised.

## Forward Method Arguments

| Argument |   Type  |                                 Description                                  |
|:--------:|:-------:|:----------------------------------------------------------------------------:|
|    x    |  Tensor  |   Input tensor for the forward function of the module   |

## Class Example

In the code below, we initialize the ClippedGELUActivation module with a min and max value and input a tensor `x`:

```python
import torch
from torch import Tensor, nn
from torch.nn.functional import gelu

from zeta.nn import ClippedGELUActivation

# Initialize the class
clipped_gelu = ClippedGELUActivation(min=-3.0, max=3.0)

# Create a tensor
x = torch.randn(3, 3)

# Pass the tensor through the module
output = clipped_gelu(x)
```

In this instance, the output tensor would have each of its elements limited to be within the range of -3.0 to 3.0, inclusively.

## Notes

While using this class be cautious of the following:
- The class does not check if the `max` argument is less than the `min` argument. Providing a `max` which is less than `min` will raise a ValueError.
- The `forward` method does not check if all elements of the input Tensor `x` are numeric. Non-numeric input may result in unexpected behavior or errors.

## References 

For additional information and further exploration about GeLU and its applications, please refer to the following resources:

1. [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
2. [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
3. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

Note: In our documentation, we provided information about the CythonGELU and its methods. The details regarding the parameters, method details, and usage examples were provided to ensure the understanding of the class and methods.
