# BitLinear Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Overview](#overview)
3. [Installation](#installation)
4. [Usage](#usage)
   1. [absmax_quantize Function](#absmax_quantize-function)
   2. [BitLinear Class](#bitlinear-class)
   3. [Examples](#examples)
5. [Additional Information](#additional-information)
6. [Conclusion](#conclusion)

---

## 1. Introduction <a name="introduction"></a>

The `BitLinear` module is a key component for implementing quantization techniques in deep learning models, particularly in Transformers. It provides a quantization layer that helps in reducing memory and computational requirements during training and inference. This documentation comprehensively explains the `BitLinear` module, its purpose, parameters, and usage.

---

## 2. Overview <a name="overview"></a>

The `BitLinear` module is designed to perform quantization on the input tensor. It is especially useful in Transformer models where memory and computational efficiency are critical. This layer quantizes the input tensor by applying binarization to the weight parameters and using the `absmax_quantize` function for quantization.

Key features and parameters of the `BitLinear` module include:
- `dim`: The dimension of the input tensor.
- `absmax_quantize` function: A function used for quantization.

By applying quantization, the `BitLinear` module helps reduce memory usage and computational complexity, making it suitable for resource-constrained environments.

---

## 3. Installation <a name="installation"></a>

Before using the `BitLinear` module, make sure you have the required dependencies installed, including PyTorch. You can install the module using pip:

```bash
pip install bitlinear
```

---

## 4. Usage <a name="usage"></a>

In this section, we'll cover how to use the `BitLinear` module effectively. It consists of two main parts: the `absmax_quantize` function and the `BitLinear` class.

### 4.1. `absmax_quantize` Function <a name="absmax_quantize-function"></a>

The `absmax_quantize` function is used to quantize a given input tensor. It follows the steps of calculating a scale, quantizing the input tensor, and dequantizing the quantized tensor.

#### Parameters:
- `x`: The input tensor to be quantized.

#### Returns:
- `quant`: The quantized tensor.
- `dequant`: The dequantized tensor.

#### Example:
```python
import torch
from zeta.quant import absmax_quantize

# Example data
x = torch.randn(10, 512)

# Quantize and dequantize
quant, dequant = absmax_quantize(x)
print(quant)
```

### 4.2. `BitLinear` Class <a name="bitlinear-class"></a>

The `BitLinear` class is the core component that implements the quantization process using binary weights. It takes the input tensor, applies normalization, binarizes the weights, performs linear operations with binarized weights, and quantizes the output.

#### Parameters:
- `dim`: The dimension of the input tensor.

#### Example:
```python
import torch
from zeta.quant import BitLinear

# Example data
x = torch.randn(10, 512)

# Initialize the BitLinear layer
layer = BitLinear(512)

# Forward pass through the BitLinear layer
y, dequant = layer(x)
print(y, dequant)
```

### 4.3. Examples <a name="examples"></a>

Let's explore three usage examples of the `BitLinear` module, demonstrating different scenarios and applications.

---

## 5. Additional Information <a name="additional-information"></a>

- **Quantization**: The `BitLinear` module is designed to perform quantization on input tensors, especially useful in resource-constrained environments and for improving efficiency in Transformer models.
- **Memory and Computational Efficiency**: It helps in reducing memory and computational requirements during training and inference.
- **Custom Quantization Functions**: You can use custom quantization functions like `absmax_quantize` to fine-tune quantization according to your requirements.

---

## 6. Conclusion <a name="conclusion"></a>

The `BitLinear` module is a valuable tool for implementing quantization in deep learning models. This documentation provides a comprehensive guide on its usage, parameters, and examples, enabling you to integrate it into your projects effectively.

Quantization plays a crucial role in optimizing models for various applications, and the `BitLinear` module simplifies this process.

*Please check the official `BitLinear` repository and documentation for any updates beyond the knowledge cutoff date.*