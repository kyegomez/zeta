# `MBConv` Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose and Functionality](#purpose-and-functionality)
3. [Function: `MBConv`](#function-mbconv)
   - [Parameters](#parameters)
4. [Usage Examples](#usage-examples)
   - [Using the `MBConv` Function](#using-the-mbconv-function)
5. [Additional Information](#additional-information)
6. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the Zeta documentation! In this documentation, we will explore the `MBConv` function, a part of the Zeta library. The `MBConv` function is designed to create MobileNetV2-like inverted residual blocks. This documentation provides a comprehensive understanding of the purpose, functionality, and usage of the `MBConv` function.

---

## 2. Purpose and Functionality <a name="purpose-and-functionality"></a>

The `MBConv` function is a building block commonly used in deep learning architectures, particularly in mobile and efficient neural networks like MobileNetV2. It creates an inverted residual block, which is characterized by a bottleneck structure that reduces the number of input channels while increasing the number of output channels. This helps reduce computational complexity while maintaining expressive power.

The key features of the `MBConv` function include:

- Configurable architecture: You can specify various parameters such as the input and output dimensions, expansion rate, shrinkage rate, and dropout rate.

- Efficient design: `MBConv` follows the design principles of MobileNetV2, making it suitable for efficient and lightweight neural network architectures.

- Flexibility: Inverted residual blocks created using `MBConv` can be used as fundamental building blocks in a variety of neural network architectures.

---

## 3. Function: `MBConv` <a name="function-mbconv"></a>

The `MBConv` function creates an inverted residual block with the following signature:

```python
def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate=4,
    shrinkage_rate=0.25,
    dropout=0.
)
```

### Parameters <a name="parameters"></a>

- `dim_in` (int): The number of input channels.

- `dim_out` (int): The number of output channels.

- `downsample` (bool): A boolean flag indicating whether downsampling should be applied. If `True`, it performs spatial downsampling. If `False`, it does not perform downsampling.

- `expansion_rate` (float, optional): The expansion rate controls the width of the bottleneck layer. It determines the number of intermediate channels in the bottleneck. Default is `4`.

- `shrinkage_rate` (float, optional): The shrinkage rate controls the reduction in the number of output channels compared to the intermediate channels. It is used in the squeeze-and-excitation (SE) block. Default is `0.25`.

- `dropout` (float, optional): The dropout rate applied to the output of the bottleneck layer. Default is `0.0`.

---

## 4. Usage Examples <a name="usage-examples"></a>

Let's explore how to use the `MBConv` function effectively in various scenarios.

### Using the `MBConv` Function <a name="using-the-mbconv-function"></a>

Here's how to use the `MBConv` function to create an inverted residual block:

```python
from zeta.nn import MBConv
import torch

# Create an inverted residual block with 64 input channels, 128 output channels, and downsampling
mbconv_block = MBConv(64, 128, downsample=True)

# Create an input tensor
x = torch.randn(32, 64, 32, 32)  # Example input with 32 samples and 64 channels

# Apply the inverted residual block
output = mbconv_block(x)

# Output tensor
print(output)
```

---

## 5. Additional Information <a name="additional-information"></a>

Inverted residual blocks, as implemented by the `MBConv` function, are widely used in efficient neural network architectures. Here are some additional notes:

- **MobileNetV2 Inspiration**: The `MBConv` function is inspired by the design of MobileNetV2, a popular mobile neural network architecture known for its efficiency.

- **Bottleneck Structure**: The use of a bottleneck structure in `MBConv` reduces computational cost while allowing the network to capture complex patterns.

- **Squeeze-and-Excitation (SE) Block**: `MBConv` includes a squeeze-and-excitation (SE) block that adaptively scales channel-wise features, enhancing the representation power of the block.

---

## 6. References <a name="references"></a>

For further information on MobileNetV2, inverted residual blocks, and related concepts, you can refer to the following resources:

- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) - The original research paper introducing MobileNetV2.

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Official PyTorch documentation for related functions and modules.

This documentation provides a comprehensive overview of the Zeta library's `MBConv` function. It aims to help you understand the purpose, functionality, and usage of the `MBConv` function for creating efficient and lightweight neural network architectures.