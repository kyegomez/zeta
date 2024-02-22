# `RMSNorm` Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose and Functionality](#purpose-and-functionality)
3. [Class: `RMSNorm`](#class-rmsnorm)
   - [Initialization](#initialization)
   - [Parameters](#parameters)
   - [Forward Method](#forward-method)
4. [Usage Examples](#usage-examples)
   - [Using the `RMSNorm` Class](#using-the-rmsnorm-class)
5. [Additional Information](#additional-information)
6. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the Zeta documentation! In this documentation, we will explore the `RMSNorm` class, a part of the Zeta library. The `RMSNorm` class is designed to perform Root Mean Square Normalization (RMSNorm) on input tensors. This documentation provides a comprehensive understanding of the purpose, functionality, and usage of the `RMSNorm` class.

---

## 2. Purpose and Functionality <a name="purpose-and-functionality"></a>

The `RMSNorm` class implements the Root Mean Square Normalization (RMSNorm) technique. RMSNorm is a normalization technique that helps stabilize the training of neural networks. It is particularly useful when dealing with deep neural networks, where gradients can vanish or explode during training.

RMSNorm works by normalizing the input tensor to have unit variance along a specified dimension, typically the feature dimension. This normalization helps prevent issues like gradient explosion and can lead to faster and more stable convergence during training.

---

## 3. Class: `RMSNorm` <a name="class-rmsnorm"></a>

The `RMSNorm` class implements the RMSNorm normalization technique. Let's dive into its details.

### Initialization <a name="initialization"></a>

To create an instance of the `RMSNorm` class, you need to specify the following parameters:

```python
RMSNorm(dim, groups=1)
```

### Parameters <a name="parameters"></a>

- `dim` (int): The dimensionality of the input tensor. This dimension will be normalized.

- `groups` (int, optional): The number of groups to divide the input tensor into before normalization. This is useful when applying RMSNorm to specific subsets of features within the input tensor. Default is `1`.

### Forward Method <a name="forward-method"></a>

The `forward` method of the `RMSNorm` class performs the RMSNorm normalization on the input tensor.

```python
def forward(x):
    # ...
    return normed * self.scale * self.gamma
```

---

## 4. Usage Examples <a name="usage-examples"></a>

Let's explore how to use the `RMSNorm` class effectively in various scenarios.

### Using the `RMSNorm` Class <a name="using-the-rmsnorm-class"></a>

Here's how to use the `RMSNorm` class to perform RMSNorm normalization on an input tensor:

```python
import torch

from zeta.nn import RMSNorm

# Create an instance of RMSNorm
rms_norm = RMSNorm(dim=512, groups=1)

# Create an input tensor
input_tensor = torch.randn(
    2, 512, 4, 4
)  # Example input tensor with shape (batch_size, channels, height, width)

# Apply RMSNorm normalization
normalized_tensor = rms_norm(input_tensor)
```

---

## 5. Additional Information <a name="additional-information"></a>

RMSNorm is a powerful technique for normalizing neural network activations during training. Here are a few additional notes:

- **Normalization Dimension (`dim`)**: The `dim` parameter specifies the dimension along which the input tensor will be normalized. It is typically set to the feature dimension (e.g., channels in a convolutional neural network).

- **Grouped Normalization (`groups`)**: The `groups` parameter allows you to divide the input tensor into groups before normalization. This can be useful when you want to apply normalization to specific subsets of features within the input tensor.

---

## 6. References <a name="references"></a>

For further information on Root Mean Square Normalization (RMSNorm) and related concepts, you can refer to the following resources:

- [Layer Normalization](https://arxiv.org/abs/1607.06450) - The original paper introducing Layer Normalization, which is a related normalization technique.

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Official PyTorch documentation for related functions and modules.

This documentation provides a comprehensive overview of the Zeta library's `RMSNorm` class. It aims to help you understand the purpose, functionality, and usage of the `RMSNorm` class for normalization in neural networks.