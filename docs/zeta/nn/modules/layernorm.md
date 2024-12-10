# `LayerNorm` Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose and Functionality](#purpose-and-functionality)
3. [Class: `LayerNorm`](#class-layernorm)
   - [Parameters](#parameters)
4. [Function: `l2norm`](#function-l2norm)
5. [Usage Examples](#usage-examples)
   - [Using the `LayerNorm` Class](#using-the-layernorm-class)
   - [Using the `l2norm` Function](#using-the-l2norm-function)
6. [Additional Information](#additional-information)
7. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the Zeta documentation! In this documentation, we will explore the `LayerNorm` class and the `l2norm` function, both of which are part of the Zeta library. These components are designed for normalization operations in neural networks. This documentation provides a comprehensive understanding of their purpose, functionality, and usage.

---

## 2. Purpose and Functionality <a name="purpose-and-functionality"></a>

The `LayerNorm` class and the `l2norm` function are essential tools in deep learning, specifically for normalizing tensors within neural networks. They offer the following functionalities:

### `LayerNorm` Class

- **Layer Normalization**: The `LayerNorm` class implements layer normalization, a technique commonly used in neural networks to stabilize training and improve generalization.

- **Configurability**: It allows you to specify the dimension for normalization and fine-tune numerical stability using parameters like `eps` and `fp16_eps`.

- **Learnable Scaling**: The class introduces a learnable scaling parameter `g` to control the magnitude of the normalized output.

### `l2norm` Function

- **L2 Normalization**: The `l2norm` function performs L2 normalization, scaling each input vector to have a unit L2 norm.

- **Tensor Normalization**: It's particularly useful when you want to normalize the magnitude of vectors or tensors in a neural network.

---

## 3. Class: `LayerNorm` <a name="class-layernorm"></a>

The `LayerNorm` class implements layer normalization with the following signature:

```python
class LayerNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps=1e-5,
        fp16_eps=1e-3,
        stable=False
    )

    def forward(self, x)
```

### Parameters <a name="parameters"></a>

- `dim` (int): The dimension of the input tensor that should be normalized.

- `eps` (float, optional): A small value added to the denominator for numerical stability when using float32 data type. Default is `1e-5`.

- `fp16_eps` (float, optional): A small value added to the denominator for numerical stability when using float16 (fp16) data type. Default is `1e-3`.

- `stable` (bool, optional): Whether to use a stable implementation of layer normalization. Default is `False`.

---

## 4. Function: `l2norm` <a name="function-l2norm"></a>

The `l2norm` function performs L2 normalization on input tensors with the following signature:

```python
def l2norm(t)
```

### Parameters

- `t` (torch.Tensor): The input tensor to be L2 normalized.

---

## 5. Usage Examples <a name="usage-examples"></a>

Let's explore how to use the `LayerNorm` class and the `l2norm` function effectively in various scenarios.

### Using the `LayerNorm` Class <a name="using-the-layernorm-class"></a>

Here's how to use the `LayerNorm` class to normalize a tensor:

```python
import torch

from zeta.nn import LayerNorm

# Create an instance of LayerNorm for a tensor with 10 dimensions
layer_norm = LayerNorm(dim=10)

# Create a random input tensor
x = torch.randn(32, 10)  # Example input with 32 samples and 10 dimensions

# Apply layer normalization
normalized_x = layer_norm(x)

# Print the normalized tensor
print(normalized_x)
```

### Using the `l2norm` Function <a name="using-the-l2norm-function"></a>

Here's how to use the `l2norm` function to perform L2 normalization on a tensor:

```python
import torch

from zeta.nn import l2norm

# Create a random input tensor
x = torch.randn(32, 10)  # Example input with 32 samples and 10 dimensions

# Apply L2 normalization
normalized_x = l2norm(x)

# Print the normalized tensor
print(normalized_x)
```

---

## 6. Additional Information <a name="additional-information"></a>

Here are some additional notes and tips related to `LayerNorm` and `l2norm`:

- **Numerical Stability**: The `eps` and `fp16_eps` parameters ensure numerical stability during normalization, especially when dealing with very small or very large values.

- **Learnable Scaling**: The learnable scaling parameter `g` in `LayerNorm` allows the model to adaptively scale the normalized output.

- **Layer Normalization**: Layer normalization is widely used in deep learning to stabilize training and improve convergence.

- **L2 Normalization**: L2 normalization is useful for scaling the magnitude of vectors or tensors to a unit L2 norm.

---

## 7. References <a name="references"></a>

For further information on layer normalization, L2 normalization, and related concepts, you can refer to the following resources:

- [Layer Normalization](https://arxiv.org/abs/1607.06450) - The original research paper introducing layer normalization.

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Official PyTorch documentation for related functions and modules.

This documentation provides a comprehensive overview of the Zeta library's `LayerNorm` class and `l2norm` function. It aims to help you understand the purpose, functionality, and usage of these components for normalization operations within neural networks.