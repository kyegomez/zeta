# `MLP` Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose and Functionality](#purpose-and-functionality)
3. [Class: `MLP`](#class-mlp)
   - [Initialization](#initialization)
   - [Parameters](#parameters)
   - [Forward Method](#forward-method)
4. [Usage Examples](#usage-examples)
   - [Using the `MLP` Class](#using-the-mlp-class)
5. [Additional Information](#additional-information)
6. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the Zeta documentation! In this documentation, we will explore the `MLP` class, a part of the Zeta library. The `MLP` class is designed to implement a Multi-Layer Perceptron (MLP) neural network module. This documentation provides a comprehensive understanding of the purpose, functionality, and usage of the `MLP` class.

---

## 2. Purpose and Functionality <a name="purpose-and-functionality"></a>

The `MLP` class implements a Multi-Layer Perceptron (MLP) module, a type of artificial neural network commonly used in deep learning. MLPs are composed of multiple layers of fully connected neurons and are known for their ability to approximate complex functions.

The key features of the `MLP` class include:

- Configurable architecture: You can specify the input and output dimensions, the expansion factor for hidden layers, the number of hidden layers, and whether to apply layer normalization.

- Activation functions: The MLP uses the Scaled Exponential Linear Unit (SiLU) activation function, which has been shown to improve training dynamics.

- Optional layer normalization: You can enable or disable layer normalization for the hidden layers.

- Flexibility: MLPs can be used for a wide range of tasks, including regression, classification, and function approximation.

---

## 3. Class: `MLP` <a name="class-mlp"></a>

The `MLP` class implements the Multi-Layer Perceptron (MLP) neural network module. Let's delve into its details.

### Initialization <a name="initialization"></a>

To create an instance of the `MLP` class, you need to specify the following parameters:

```python
MLP(
    dim_in,
    dim_out,
    *,
    expansion_factor=2.,
    depth=2,
    norm=False
)
```

### Parameters <a name="parameters"></a>

- `dim_in` (int): The dimensionality of the input tensor.

- `dim_out` (int): The dimensionality of the output tensor.

- `expansion_factor` (float, optional): The expansion factor for the hidden dimension. Default is `2.0`.

- `depth` (int, optional): The number of hidden layers. Default is `2`.

- `norm` (bool, optional): Whether to apply layer normalization to the hidden layers. Default is `False`.

### Forward Method <a name="forward-method"></a>

The `forward` method of the `MLP` class performs the forward pass of the MLP module. It takes an input tensor and returns the output tensor.

```python
def forward(x):
    """
    Forward pass of the MLP module.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor.

    """
    return self.net(x.float())
```

---

## 4. Usage Examples <a name="usage-examples"></a>

Let's explore how to use the `MLP` class effectively in various scenarios.

### Using the `MLP` Class <a name="using-the-mlp-class"></a>

Here's how to use the `MLP` class to create and apply an MLP neural network:

```python
import torch

from zeta.nn import MLP

# Create an instance of MLP
mlp = MLP(dim_in=256, dim_out=10, expansion_factor=4.0, depth=3, norm=True)

# Create an input tensor
x = torch.randn(32, 256)

# Apply the MLP
output = mlp(x)

# Output tensor
print(output)
```

---

## 5. Additional Information <a name="additional-information"></a>

Multi-Layer Perceptrons (MLPs) are versatile neural network architectures that can be adapted to various tasks. Here are some additional notes:

- **Hidden Layer Configuration**: You can customize the architecture of the MLP by adjusting parameters such as `expansion_factor` and `depth`. These parameters control the size and depth of the hidden layers.

- **Layer Normalization**: Layer normalization can help stabilize training and improve convergence, especially in deep networks. You can enable it by setting the `norm` parameter to `True`.

- **Activation Function**: The MLP uses the Scaled Exponential Linear Unit (SiLU) activation function, which is known for its smooth gradients and improved training dynamics.

---

## 6. References <a name="references"></a>

For further information on Multi-Layer Perceptrons (MLPs) and related concepts, you can refer to the following resources:

- [Deep Learning](https://www.deeplearningbook.org/) - "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. This book provides an in-depth understanding of neural networks, including MLPs.

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Official PyTorch documentation for related functions and modules.

This documentation provides a comprehensive overview of the Zeta library's `MLP` class. It aims to help you understand the purpose, functionality, and usage of the `MLP` class for building Multi-Layer Perceptron neural networks for various tasks.