# `CustomMLP`

## Introduction

Welcome to the documentation for `zeta.nn`! This module provides a customizable Multi-Layer Perceptron (MLP) implementation using PyTorch. With `CustomMLP`, you can create and configure your own MLP architecture for various machine learning tasks. This documentation will guide you through the functionalities, usage, and customization options of `CustomMLP`.

## Table of Contents

1. [Installation](#installation)
2. [Overview](#overview)
3. [Class Definition](#class-definition)
4. [Functionality and Usage](#functionality-and-usage)
    - [Initialization](#initialization)
    - [Forward Pass](#forward-pass)
    - [Customization](#customization)
5. [Examples](#examples)
6. [Additional Information](#additional-information)
7. [References](#references)

## 1. Installation <a name="installation"></a>

Before using `CustomMLP`, make sure you have `zetascale` installed. You can install it using:

```bash
pip install zetascale
```

Once PyTorch is installed, you can import `CustomMLP` from `zeta.nn` as follows:

```python
from zeta.nn import CustomMLP
```

## 2. Overview <a name="overview"></a>

`CustomMLP` is a versatile MLP architecture that allows you to define the number of layers, layer sizes, activation functions, and dropout probability according to your specific requirements. It is suitable for tasks like classification, regression, and more.

Key features:
- Customizable layer sizes and activation functions.
- Dropout regularization for improved generalization.
- Supports popular activation functions like ReLU, Sigmoid, and Tanh.

## 3. Class Definition <a name="class-definition"></a>

### `CustomMLP`

```markdown
| Attribute          | Description                                            |
|--------------------|--------------------------------------------------------|
| layers             | List of linear layers.                                 |
| activation_fn      | Activation function to be applied after each layer.   |
| dropout            | Dropout probability for regularization.               |

Parameters:
- `layer_sizes` (list of int): List of layer sizes including input and output layer.
- `activation` (str, optional): Type of activation function. Default is 'relu'.
- `dropout` (float, optional): Dropout probability. Default is 0.0 (no dropout).
```

## 4. Functionality and Usage <a name="functionality-and-usage"></a>

### Initialization <a name="initialization"></a>

To create an instance of `CustomMLP`, you need to specify the `layer_sizes`, which is a list of integers representing the sizes of each layer, including the input and output layers. You can also customize the `activation` function and `dropout` probability.

Example:

```python
from zeta.nn import CustomMLP

# Create an MLP with 3 layers: input (10), hidden (5), and output (2)
mlp = CustomMLP(layer_sizes=[10, 5, 2], activation='relu', dropout=0.5)
```

### Forward Pass <a name="forward-pass"></a>

You can perform a forward pass through the MLP by passing input data to it. The input data should be a PyTorch tensor.

Example:

```python
import torch

# Input data (1 sample with 10 features)
input_data = torch.randn(1, 10)

# Forward pass through the MLP
output = mlp(input_data)
```

### Customization <a name="customization"></a>

You can customize the following aspects of the MLP:
- **Layer Sizes**: Specify the sizes of layers in the `layer_sizes` parameter.
- **Activation Function**: Choose from 'relu' (default), 'sigmoid', or 'tanh' for activation.
- **Dropout**: Set the `dropout` probability for regularization.

## 5. Examples <a name="examples"></a>

### Example 1: Customizing MLP

```python
from zeta.nn import CustomMLP

# Create an MLP with custom layer sizes, sigmoid activation, and dropout
mlp = CustomMLP(layer_sizes=[20, 10, 5], activation='sigmoid', dropout=0.2)
```

### Example 2: Forward Pass

```python
import torch
from zeta.nn import CustomMLP

# Define the layer sizes
layer_sizes = [5, 10, 1]

# Create the MLP
mlp = CustomMLP(layer_sizes, activation="relu", dropout=0.5)

# Create a random tensor of shape (batch_size, input_size)
x = torch.randn(32, 5)

# Pass the tensor through the MLP
output = mlp(x)

print(output)
```

### Example 3: Customizing and Forward Pass

```python
import torch
from zeta.nn import CustomMLP

# Create an MLP with custom configuration
mlp = CustomMLP(layer_sizes=[15, 8, 3], activation='tanh', dropout=0.3)

# Input data (single sample with 15 features)
input_data = torch.randn(1, 15)

# Forward pass through the customized MLP
output = mlp(input_data)
```

## 6. Additional Information <a name="additional-information"></a>

- If you encounter any issues or have questions, please refer to the [References](#references) section for further resources.

## 7. References <a name="references"></a>

- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- PyTorch Tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

This concludes the documentation for `zeta.nn` and the `CustomMLP` class. You are now equipped to create and customize your MLP architectures for various machine learning tasks. Happy coding!