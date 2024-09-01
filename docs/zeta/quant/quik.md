# QUIK: Quantized Integers with Kernels (QUIK) for Efficient Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture](#architecture)
    1. [QUIK Layer](#quik-layer)
    2. [Quantization](#quantization)
    3. [Dequantization](#dequantization)
3. [Installation](#installation)
4. [Usage](#usage)
    1. [Initializing QUIK Layer](#initializing-quik-layer)
    2. [Quantizing Data](#quantizing-data)
    3. [Dequantizing Data](#dequantizing-data)
    4. [Forward Pass](#forward-pass)
5. [Examples](#examples)
    1. [Example 1: Initializing QUIK Layer](#example-1-initializing-quik-layer)
    2. [Example 2: Quantizing Data](#example-2-quantizing-data)
    3. [Example 3: Dequantizing Data](#example-3-dequantizing-data)
    4. [Example 4: Forward Pass](#example-4-forward-pass)
6. [Additional Information](#additional-information)
7. [Conclusion](#conclusion)

---

## 1. Introduction <a name="introduction"></a>

**QUIK (Quantized Integers with Kernels)** is a PyTorch-based module designed to enable efficient deep learning by leveraging quantization techniques. This module provides a custom QUIK layer that performs quantization and dequantization operations on input data. It's particularly useful when memory and computation resources are constrained, making it suitable for deployment on edge devices.

The key features of QUIK include:
- Quantization of input data to a reduced bit-width.
- Efficient kernel operations on quantized data.
- Dequantization of results back to floating-point values.

In this documentation, we'll explore the architecture of the QUIK module, how to install it, and provide detailed examples of its usage.

---

## 2. Architecture <a name="architecture"></a>

The QUIK module consists of a custom QUIK layer, which performs quantization, efficient kernel operations, and dequantization. Let's dive into the architecture in detail.

### 2.1. QUIK Layer <a name="quik-layer"></a>

The QUIK layer is the core component of the module. It takes input data and performs quantization, efficient weighted summation, and dequantization.

### 2.2. Quantization <a name="quantization"></a>

Quantization is the process of converting input data to a reduced bit-width representation. In the case of QUIK, it quantizes the input tensor to an integer representation, typically using fewer bits than a standard floating-point representation.

### 2.3. Dequantization <a name="dequantization"></a>

Dequantization is the inverse process of quantization. It takes quantized data and converts it back to a floating-point representation. This operation ensures that the final output is in a format suitable for further computations or analysis.

---

## 3. Installation <a name="installation"></a>

You can install the QUIK module via pip. Open your terminal and run the following command:

```bash
pip install quik
```

---

## 4. Usage <a name="usage"></a>

Let's explore how to use the QUIK module step by step.

### 4.1. Initializing QUIK Layer <a name="initializing-quik-layer"></a>

First, you need to initialize the QUIK layer by specifying the number of input and output features. Optionally, you can enable bias terms. Here's how to do it:

```python
import torch
import torch.nn as nn

# Initialize the QUIK module
quik = QUIK(in_features=784, out_features=10, bias=True)
```

### 4.2. Quantizing Data <a name="quantizing-data"></a>

You can quantize your input data using the `quantize` method of the QUIK layer. This method returns the quantized data, zero-point, and scale factor.

```python
# Create some dummy data, e.g., simulating a batch of MNIST images
data = torch.randn(10, 784)

# Quantize the data
quantized_data, zero_point, scale_factor = quik.quantize(data)
```

### 4.3. Dequantizing Data <a name="dequantizing-data"></a>

To dequantize data, use the `dequantize` method of the QUIK layer. This method requires the quantized data, zero-point, scale factor, and an additional scale factor for weights.

```python
# Dequantize the quantized data
dequantized_data = quik.dequantize(
    quantized_data, zero_point, scale_factor, scale_weight
)
```

### 4.4. Forward Pass <a name="forward-pass"></a>

Now, you can run the quantized data through the QUIK layer to perform the forward pass. This will quantize the data, apply the weight operation efficiently, and dequantize the result.

```python
# Forward pass
output = quik(quantized_data)
```

---

## 5. Examples <a name="examples"></a>

Let's go through some examples to illustrate how to use the QUIK module effectively.

### 5.1. Example 1: Initializing QUIK Layer <a name="example-1-initializing-quik-layer"></a>

In this example, we'll initialize the QUIK layer.

```python
import torch

from zeta.nn.quant import QUIK

# Initialize the QUIK module
quik = QUIK(in_features=784, out_features=10)
```

### 5.2. Example 2: Quantizing Data <a name="example-2-quantizing-data"></a>

Now, we'll quantize some input data.

```python
# Create some dummy data, e.g., simulating a batch of MNIST images
data = torch.randn(10, 784)

# Quantize the data
quantized_data, zero_point, scale_factor = quik.quantize(data)
```

### 5.3. Example 3: Dequantizing Data <a name="example-3-dequantizing-data"></a>

In this example, we'll dequantize the quantized data.

```python
# Dequantize the quantized data
dequantized_data = quik.dequantize(
    quantized_data, zero_point, scale_factor, scale_weight
)
```

### 5.4. Example 4: Forward Pass <a name="example-4-forward-pass"></a>

Finally, we'll perform a forward pass using the QUIK layer.

```python
# Forward pass
output = quik(quantized_data)
```

---

## 6. Additional Information <a name="additional-information"></a>

- **Performance**: QUIK is designed for efficient deep learning, especially in resource-constrained environments.
- **Quantization Range**: By default, QUIK assumes 4-bit quantization, so the range is [-8, 7].
- **Training**: QUIK is primarily intended for inference. It is not designed for training.
- **Customization**: You can customize the quantization

 range, bit-width, and other parameters as needed.

---

## 7. Conclusion <a name="conclusion"></a>

The QUIK module offers a straightforward way to apply quantization techniques to your deep learning models, making them more memory and computationally efficient. By following the guidelines and examples in this documentation, you can effectively integrate QUIK into your projects, especially when deploying models to edge devices or resource-constrained environments.

For further information and detailed function descriptions, please refer to the QUIK module's official documentation.
