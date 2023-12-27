# NewGELUActivation

# Chapter 1: Introduction and Overview

# NewGELUActivation

The NewGELUActivation class is an implementation of the Gaussian Error Linear Units (GELU) activation function. In PyTorch, activation functions are essential non-linear transformations that are applied on the input, typically after linear transformations, to introduce non-linearity into the model. The GELU activation function is currently being used in Google's BERT and OpenAI's GPT models. If you are interested in more details about this function, see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

# Chapter 2: Detailed Explanation of the NewGELUActivation Class

The `NewGELUActivation` class extends `nn.Module`, so it can be integrated easily into any PyTorch model. It is a type of activation function that is believed to perform better in deeper architectures.

```
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )
```

## Forward Function

The `forward` method **overloads** the call to the function to process data. The forward method takes one mandatory argument:

- `input` - This is a tensor that represents the activations output from the previous layer. The data type is Tensor.

The forward method returns: 

- The value obtained after applying the New GELU activation function on the input tensor.

#### Implementation of the forward method:
The forward method calculates the New GELU activation of the input tensor. The formula for calculating the New GELU activation is as follows:

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

where,
- `x` is the input.
- `tanh` is the hyperbolic tangent function.
- `sqrt` is the square root function.
- `^` is the power operator.

Importantly, when the `forward` function is called on an object of the class `NewGELUActivation`, it computes these operations on the input tensor, and the result is returned.

# Chapter 3: Usage Examples

At first, you need to import necessary packages and modules. 

```python
import torch
import math
from torch import Tensor
from torch import nn
from zeta.nn import NewGELUActivation 
```

## Usage Example 1:

Creating an instance of NewGELUActivation and calling it with a tensor as input.

```python
gelu_new = NewGELUActivation()

random_data = torch.randn(5)  # Just some random data
output = gelu_new(random_data)

print(output)
```

## Usage Example 2:

Integrating NewGELUActivation within a neural network model.

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.new_gelu = NewGELUActivation()

    def forward(self, x):
        x = self.fc1(x)
        x = self.new_gelu(x)
        return x

model = NeuralNetwork()  # Creating an instance of our model
```

## Usage Example 3:

Applying the NewGELUActivation function in a Convolutional Neural Network (CNN).

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.new_gelu = NewGELUActivation()

    def forward(self, x):
        x = self.new_gelu(self.conv1(x))
        return x

model = CNN()  # Creating an instance of our model
```

# Chapter 4: Conclusion

This was a complete guide about the `NewGELUActivation` PyTorch class. This tool provides an implementation of the GELU activation function, improving deep learning model architectures. This document demonstrated how to use the `NewGELUActivation` class and integrate it into existing PyTorch models with various examples.

# External Links

- Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415 
- PyTorch official documentation: https://pytorch.org/docs/stable/index.html 
- Other relevant resources: https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
