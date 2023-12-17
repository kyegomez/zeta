# `niva`

## Overview

The Niva module provides functionality for quantizing PyTorch neural network models, enabling you to reduce their memory and computation requirements while preserving their accuracy. Quantization is a crucial technique for deploying models on resource-constrained devices such as edge devices and mobile platforms.

This documentation will guide you through the Niva module's architecture, purpose, functions, and usage examples. You'll learn how to effectively quantize your PyTorch models and optimize their performance for different deployment scenarios.

## Table of Contents

1. [Installation](#installation)
2. [Architecture](#architecture)
3. [Purpose](#purpose)
4. [Function: niva](#function-niva)
    - [Parameters](#parameters)
    - [Usage Examples](#usage-examples)
        - [Dynamic Quantization](#dynamic-quantization)
        - [Static Quantization](#static-quantization)
5. [Additional Information](#additional-information)
6. [References](#references)

---

## 1. Installation <a name="installation"></a>

Before using the Niva module, make sure you have PyTorch installed. You can install PyTorch using the following command:

```bash
pip install zetascale
```

## 2. Architecture <a name="architecture"></a>

The Niva module leverages PyTorch's quantization capabilities to quantize neural network models. It offers both dynamic and static quantization options to accommodate various use cases.

## 3. Purpose <a name="purpose"></a>

The primary purpose of the Niva module is to enable quantization of PyTorch models. Quantization is the process of reducing the precision of model weights and activations, which results in smaller model sizes and faster inference on hardware with limited resources. This is especially important for deploying models on edge devices and mobile platforms.

## 4. Function: niva <a name="function-niva"></a>

The `niva` function is the core of the Niva module, responsible for quantizing a given PyTorch model. It supports both dynamic and static quantization modes, allowing you to choose the most suitable quantization approach for your model.

### Parameters <a name="parameters"></a>

The `niva` function accepts the following parameters:

- `model` (nn.Module): The PyTorch model to be quantized.
- `model_path` (str, optional): The path to the pre-trained model's weights. Defaults to None.
- `output_path` (str, optional): The path where the quantized model will be saved. Defaults to None.
- `quant_type` (str, optional): The type of quantization to be applied, either "dynamic" or "static". Defaults to "dynamic".
- `quantize_layers` (Union[List[Type[nn.Module]], None], optional): A list of layer types to be quantized. Defaults to None.
- `dtype` (torch.dtype, optional): The target data type for quantization, either torch.qint8 or torch.quint8. Defaults to torch.qint8.
- `*args` and `**kwargs`: Additional arguments for PyTorch's quantization functions.

### Usage Examples <a name="usage-examples"></a>

#### Dynamic Quantization <a name="dynamic-quantization"></a>

In dynamic quantization, you specify the layers to be quantized, and the quantization process occurs dynamically during inference. Here's an example:

```python
import torch
from zeta import niva

# Load a pre-trained model
model = YourModelClass()

# Quantize the model dynamically, specifying layers to quantize
niva(
    model=model,
    model_path="path_to_pretrained_model_weights.pt",
    output_path="quantized_model.pt",
    quant_type="dynamic",
    quantize_layers=[nn.Linear, nn.Conv2d],
    dtype=torch.qint8
)
```

#### Static Quantization <a name="static-quantization"></a>

Static quantization quantizes the entire model before inference. Here's an example:

```python
import torch
from zeta import niva

# Load a pre-trained model
model = YourModelClass()

# Quantize the entire model statically
niva(
    model=model,
    model_path="path_to_pretrained_model_weights.pt",
    output_path="quantized_model.pt",
    quant_type="static",
    dtype=torch.qint8
)
```

## 5. Additional Information <a name="additional-information"></a>

- The Niva module supports both dynamic and static quantization modes, giving you flexibility in choosing the right approach for your deployment scenario.
- Always ensure that your model is in evaluation mode (`model.eval()`) before quantization.
- Quantization reduces model size and inference time but may slightly affect model accuracy. It's essential to evaluate the quantized model's performance before deployment.

## 6. References <a name="references"></a>

For more information on PyTorch quantization and best practices, refer to the official PyTorch documentation: [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html).

