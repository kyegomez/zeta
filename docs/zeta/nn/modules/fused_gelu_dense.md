# `FusedDenseGELUDense`

## Overview

The `FusedDenseGELUDense` module is a versatile neural network layer designed for efficient computation of dense layers with GELU (Gaussian Error Linear Unit) activations. This documentation will provide an in-depth understanding of the module's architecture, purpose, parameters, and usage examples.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Purpose](#purpose)
4. [Class Definition](#class-definition)
    - [Parameters](#parameters)
    - [Internal Layers](#internal-layers)
5. [Functionality and Usage](#functionality-and-usage)
    - [Forward Pass](#forward-pass)
6. [Examples](#examples)
    - [Basic Usage](#basic-usage)
    - [Custom Configuration](#custom-configuration)
    - [Quantization with bitsandbytes](#quantization-with-bitsandbytes)
7. [Additional Information](#additional-information)
8. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

The `FusedDenseGELUDense` module combines dense layers with GELU activations in a single neural network layer. This fusion improves computational efficiency and is particularly useful in various deep learning applications.

## 2. Architecture <a name="architecture"></a>

The `FusedDenseGELUDense` layer consists of two dense sub-layers, each followed by a GELU activation function. It takes an input tensor and passes it through these sub-layers to produce the final output.

## 3. Purpose <a name="purpose"></a>

The primary purpose of the `FusedDenseGELUDense` layer is to efficiently compute dense transformations with GELU activations. It is designed for use in neural networks, providing a convenient way to incorporate these operations into deep learning models.

## 4. Class Definition <a name="class-definition"></a>

### Parameters <a name="parameters"></a>

- `dim` (int): Input dimension.
- `dim_out` (int): Output dimension.
- `bias` (bool, optional): Whether to include bias terms. Defaults to True.
- `has_fp16_weights` (bool, optional): Whether to use fp16 weights. Defaults to False.
- `threshold` (float, optional): Threshold for quantization. Defaults to 6.0.

### Internal Layers <a name="internal-layers"></a>

The `FusedDenseGELUDense` layer consists of the following internal layers:

1. `dense1`: The first dense layer.
2. `act`: The GELU activation function.
3. `dense2`: The second dense layer.

## 5. Functionality and Usage <a name="functionality-and-usage"></a>

### Forward Pass <a name="forward-pass"></a>

The `forward` method of the `FusedDenseGELUDense` layer performs the following operations:

1. Applies the first dense layer (`dense1`) to the input tensor.
2. Applies the GELU activation function (`act`) to the result.
3. Applies the second dense layer (`dense2`) to the GELU-activated output.

## 6. Examples <a name="examples"></a>

### Basic Usage <a name="basic-usage"></a>

Here's a basic example of using the `FusedDenseGELUDense` layer:

```python
import torch

from zeta.nn import FusedDenseGELUDense

# Create an instance of FusedDenseGELUDense
model = FusedDenseGELUDense(dim=512, dim_out=1024)

# Generate random input tensor
x = torch.randn(1, 512)

# Forward pass
out = model(x)

# Check the output shape
print(out.shape)  # torch.Size([1, 512])
```

### Custom Configuration <a name="custom-configuration"></a>

You can customize the layer by specifying different parameters:

```python
# Create a custom FusedDenseGELUDense layer
custom_model = FusedDenseGELUDense(
    dim=256, dim_out=512, bias=False, has_fp16_weights=True, threshold=4.0
)

# Generate random input tensor
x = torch.randn(1, 256)

# Forward pass with the custom configuration
out = custom_model(x)
```

### Quantization with bitsandbytes <a name="quantization-with-bitsandbytes"></a>

You can enable quantization using the `bitsandbytes` library by providing a quantized implementation of the dense layers:

```python
# Install bitsandbytes if not already installed
# pip install bitsandbytes

import torch

from zeta.nn import FusedDenseGELUDense

# Create an instance of FusedDenseGELUDense with quantization
quantized_model = FusedDenseGELUDense(
    dim=512, dim_out=1024, has_fp16_weights=True, threshold=4.0
)

# Generate random input tensor
x = torch.randn(1, 512)

# Forward pass with quantization
out = quantized_model(x)
```

## 7. Additional Information <a name="additional-information"></a>

- The `FusedDenseGELUDense` layer efficiently combines dense and GELU activation operations.
- Custom configurations for bias, weight precision, and threshold are supported.
- Quantization can be enabled using the `bitsandbytes` library for further efficiency.

## 8. References <a name="references"></a>

For more information on GELU activations and dense layers in PyTorch, refer to the official PyTorch documentation:

- [GELU Activation Function](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
- [Dense Layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
