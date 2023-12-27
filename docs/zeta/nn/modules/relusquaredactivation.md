# ReLUSquaredActivation

## Overview

The `ReLUSquaredActivation` class is a PyTorch neural network module that implements a custom activation function known as ReLU². This activation function is introduced in the [What You See Is What You Get](https://arxiv.org/abs/2109.08668v2) paper by Kim, Y., & Bengio, S., and they prove it to be an important enhancement in the stability of Neural Network Training.

This activation layer applies the ReLU (Rectified Linear Unit) function to the input and then squares the result. Thus, it can only result in non-negative outputs. The squaring operation increases the emphasis on positive inputs and reduces the effect of small inputs, aiding in reducing the outliers effect and better focusing the network on meaningful inputs.

## Class Definition

```python
class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward(self, input):
        relu_applied = nn.functional.relu(input)
        squared = torch.square(relu_applied)
        return squared
```

### `class ReLUSquaredActivation`

This is the class constructor that creates an instance of the `ReLUSquaredActivation` class.

The `ReLUSquaredActivation` class extends [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), the base class for all neural network modules in PyTorch. It does not accept any parameters.

### `forward(self, input)`

This is the forward pass of the ReLUSquaredActivation module. It's where the computation happens. This method does not have to be explicitly called, and it can be run by calling the instance of the class. 

| Argument |  Type  | Description  |
|----------|:------|:-------------|
| `input`    | Tensor | The input tensor on which the relu squared operation is to be applied.

It applies the `ReLU` activation function on the input tensor and then squares the result. It returns a tensor with the same shape as the input tensor, with the ReLU² activation applied.


## Example Usage

```python
# Importing the essential libraries
import torch
import torch.nn as nn
from zeta.nn import ReLUSquaredActivation

# Creating random torch tensor for input
input_tensor = torch.randn((2,2))

# Creating an instance of module
relu_squared_activation = ReLUSquaredActivation()

# Applying the module to input tensor
output_tensor = relu_squared_activation(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("Output Tensor:")
print(output_tensor)
```

In this example, we first import the necessary libraries. We then create an instance of `ReLUSquaredActivation`. After creating this instance, you can use it as a function to apply the ReLU² activation to the input tensor. 

In the resulting output tensor, the activation function is applied elementwise, meaning that every single value in the tensor has the activation function applied independently. This means that the shape of the output tensor is identical to the shape of the input tensor.

## Additional Information

The `ReLUSquaredActivation` is a simple yet powerful activation layer that can provide increased performance in certain types of neural networks. However, like all tools, it is important to use it in the right context and understand that it might not always lead to the best results depending on the specific problem and data at hand.

Note that the `ReLUSquaredActivation` extends the `nn.Module` class, which is the fundamental building block in PyTorch. It forms part of a larger toolkit for building and running neural networks, and there are many other types of modules available in the [`torch.nn`](https://pytorch.org/docs/stable/nn.html) library that you might find useful.
