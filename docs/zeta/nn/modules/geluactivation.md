# GELUActivation

## Overview

The GELUActivation class belongs to the torch.nn Module and implements the Gaussian Error Linear Units (GELU) activation function, initially used in Google's BERT model. This function is known for enabling the model to converge much faster and provides more robust performance in terms of model stability and accuracy.

The GELU activation function is defined as follows: 
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))

There are two versions of this function which are slightly different. The standard one implemented in PyTorch, and the original version used in the BERT model. This class provides the flexibility to choose between these two implementations.

## Class Definition

class GELUActivation(nn.Module):

This class inherits the torch.nn.Module, torch's base class for all neural network modules. 

### Parameters

- use_gelu_python (bool): If true, uses the original GELU activation function as introduced in the BERT model. Otherwise, it uses the PyTorch's implementation of GELU. Default is `False`.

### Methods

#### \_\_init__()

The constructor method for the class. Initializes the GELUActivation with the given parameters.

#### _gelu_python()

This private method implements the original GELU activation function used in the BERT model as a simple python function.

#### forward()

This method is called when you call the object of the class. It takes an input tensor and applies the GELU activation function to it.

## Usage Example

Here is an example usage of the GELUActivation class. The example demonstrates initializing the class and applying the GELU activation function to a random tensor.

```python
import torch
from torch import Tensor, nn

from zeta.nn import GELUActivation

# Initialize a GELU activation function
gelu_activation = GELUActivation(use_gelu_python=True)

# Generate a random tensor
tensor = torch.randn(5)

# Apply GELU activation function to the tensor
activated_tensor = gelu_activation(tensor)

print(activated_tensor)
```

In this example, we initialize a GELU activation function with `use_gelu_python` set to `True` which means we will be using the original GELU implementation used in the BERT model. We then apply this GELU activation function to a random tensor to get the activated tensor.

## References

- Gaussian Error Linear Units (GELUs) Paper: [https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)

We suggest to read the referenced paper to gain a deeper understanding of GELUs and their use in neural networks.

## Tips and Tricks

- While the two versions of the GELU activation function are very similar, the original one (used in the BERT model) can sometimes provide slightly different results.
- If you're using a model pre-trained with the BERT model, it may be beneficial to use the original version of GELU, as it was the activation functions that the model was originally trained with.
- GELU activation function has proven effective in models dealing with Natural Language Processing tasks.
