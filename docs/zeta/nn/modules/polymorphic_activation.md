# `PolymorphicNeuronLayer` Documentation

## Introduction

Welcome to the documentation for `zeta.nn`! This module provides a unique and versatile Polymorphic Neuron Layer implemented using PyTorch. The `PolymorphicNeuronLayer` is designed to introduce dynamic activation functions within a neural network layer, allowing for adaptive learning. This documentation aims to comprehensively explain the purpose, architecture, usage, and customization options of the `PolymorphicNeuronLayer`.

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

Before using `PolymorphicNeuronLayer`, make sure you have `zetascale` installed. You can install it using:

```bash
pip install zetascale
```

Once PyTorch is installed, you can import `PolymorphicNeuronLayer` from `zeta.nn` as follows:

```python
from zeta.nn import PolymorphicNeuronLayer
```

## 2. Overview <a name="overview"></a>

The `PolymorphicNeuronLayer` is a groundbreaking neural network layer that introduces dynamic activation functions to each neuron within the layer. This unique approach enables neurons to adapt and select activation functions based on their input data, leading to more flexible and adaptive learning.

Key features:
- Adaptive activation functions per neuron.
- Customizable input and output features.
- Support for multiple activation functions.

## 3. Class Definition <a name="class-definition"></a>

### `PolymorphicNeuronLayer`

```
| Attribute                  | Description                                            |
|----------------------------|--------------------------------------------------------|
| in_features                 | Number of input features.                              |
| out_features                | Number of output features (neurons).                   |
| activation_functions        | List of activation functions to choose from.           |
| weights                    | Learnable weights for linear transformation.           |
| bias                       | Learnable bias term.                                   |

Parameters:
- `in_features` (int): Number of input features.
- `out_features` (int): Number of output features (neurons).
- `activation_functions` (list of callable): List of activation functions to choose from.
```

## 4. Functionality and Usage <a name="functionality-and-usage"></a>

### Initialization <a name="initialization"></a>

To create an instance of `PolymorphicNeuronLayer`, you need to specify the `in_features`, `out_features`, and provide a list of `activation_functions`. These activation functions will be used dynamically based on neuron-specific criteria.

Example:

```python
import torch.nn.functional as F

from zeta.nn import PolymorphicNeuronLayer

# Create a Polymorphic Neuron Layer with 10 input features, 5 output neurons, and a list of activation functions
neuron = PolymorphicNeuronLayer(
    in_features=10, out_features=5, activation_functions=[F.relu, F.tanh, F.sigmoid]
)
```

### Forward Pass <a name="forward-pass"></a>

You can perform a forward pass through the `PolymorphicNeuronLayer` by passing input data to it. The input data should be a PyTorch tensor.

Example:

```python
import torch

# Input data (1 sample with 10 features)
input_data = torch.randn(1, 10)

# Forward pass through the Polymorphic Neuron Layer
output = neuron(input_data)
```

### Customization <a name="customization"></a>

You can customize the following aspects of the `PolymorphicNeuronLayer`:
- **Input Features**: Set the number of input features in the `in_features` parameter.
- **Output Features**: Set the number of output neurons in the `out_features` parameter.
- **Activation Functions**: Provide a list of activation functions to choose from in `activation_functions`.

## 5. Examples <a name="examples"></a>

### Example 1: Customizing and Forward Pass

```python
import torch.nn.functional as F

from zeta.nn import PolymorphicNeuronLayer

# Create a Polymorphic Neuron Layer with custom configuration
neuron = PolymorphicNeuronLayer(
    in_features=15, out_features=8, activation_functions=[F.relu, F.tanh, F.sigmoid]
)

# Input data (single sample with 15 features)
input_data = torch.randn(1, 15)

# Forward pass through the customized Polymorphic Neuron Layer
output = neuron(input_data)
```

### Example 2: Custom Activation Functions

```python
from zeta.nn import PolymorphicNeuronLayer


# Define custom activation functions
def custom_activation_1(x):
    return x**2


def custom_activation_2(x):
    return torch.sin(x)


# Create a Polymorphic Neuron Layer with custom activation functions
neuron = PolymorphicNeuronLayer(
    in_features=5,
    out_features=3,
    activation_functions=[custom_activation_1, custom_activation_2],
)

# Input data (1 sample with 5 features)
input_data = torch.randn(1, 5)

# Forward pass through the Polymorphic Neuron Layer with custom activations
output = neuron(input_data)
```

### Example 3: Dynamic Activation Selection

```python
import torch.nn.functional as F

from zeta.nn import PolymorphicNeuronLayer

# Create a Polymorphic Neuron Layer with 5 input features, 3 output neurons, and standard activation functions
neuron = PolymorphicNeuronLayer(
    in_features=5, out_features=3, activation_functions=[F.relu, F.tanh, F.sigmoid]
)

# Input data (single sample with 5 features)
input_data = torch.randn(1, 5)

# Forward pass through the Polymorphic Neuron Layer with dynamic activation selection
output = neuron(input_data)
```

## 6. Additional Information <a name="additional-information"></a>

- The dynamic activation selection in the `PolymorphicNeuronLayer` enhances adaptability and learning capacity within neural networks.
- For more advanced use cases and custom activation functions, you can define your own callable functions and pass them to the layer.

## 7. References <a name="references"></a>

- PyTorch Documentation

: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- PyTorch Tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

This concludes the documentation for `zeta.nn` and the `PolymorphicNeuronLayer` class. You now have the knowledge to incorporate dynamic activation functions into your neural networks for more adaptive and flexible learning. Happy coding!