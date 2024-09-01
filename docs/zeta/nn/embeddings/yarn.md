# `YarnEmbedding` Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose and Functionality](#purpose-and-functionality)
3. [Class: `YarnEmbedding`](#class-yarnembedding)
   - [Initialization](#initialization)
   - [Parameters](#parameters)
   - [Forward Method](#forward-method)
4. [Helpers and Functions](#helpers-and-functions)
   - [`find_correction_dim`](#find-correction-dim)
   - [`find_correction_range`](#find-correction-range)
   - [`linear_ramp_mask`](#linear-ramp-mask)
   - [`get_mscale`](#get-mscale)
5. [Usage Examples](#usage-examples)
   - [Using the `YarnEmbedding` Class](#using-the-yarnembedding-class)
   - [Using the Helper Functions](#using-the-helper-functions)
6. [Additional Information](#additional-information)
   - [Positional Embeddings in Transformers](#positional-embeddings-in-transformers)
7. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the Zeta documentation for the `YarnEmbedding` class and related functions! Zeta is a powerful library for deep learning in PyTorch, and this documentation will provide a comprehensive understanding of the `YarnEmbedding` class and its associated functions.

---

## 2. Purpose and Functionality <a name="purpose-and-functionality"></a>

The `YarnEmbedding` class and its related functions are designed to generate and apply advanced positional embeddings to input tensors. These embeddings are crucial for sequence-to-sequence models, particularly in transformer architectures. Below, we will explore their purpose and functionality.

---

## 3. Class: `YarnEmbedding` <a name="class-yarnembedding"></a>

The `YarnEmbedding` class is used to apply advanced positional embeddings to input tensors. It offers a highly configurable approach to generating embeddings tailored to the needs of transformer-based models.

### Initialization <a name="initialization"></a>

To create an instance of the `YarnEmbedding` class, you need to specify the following parameters:

```python
YarnEmbedding(
    dim,
    max_position_embeddings=2048,
    base=10000,
    original_max_position_embeddings=2048,
    extrapolation_factor=1,
    attn_factor=1,
    beta_fast=32,
    beta_slow=1,
    finetuned=False,
    device=None,
)
```

### Parameters <a name="parameters"></a>

- `dim` (int): The dimensionality of the positional embeddings.

- `max_position_embeddings` (int, optional): The maximum number of position embeddings to be generated. Default is `2048`.

- `base` (int, optional): The base value for calculating the positional embeddings. Default is `10000`.

- `original_max_position_embeddings` (int, optional): The original maximum number of position embeddings used for fine-tuning. Default is `2048`.

- `extrapolation_factor` (int, optional): The factor used for extrapolating positional embeddings beyond the original maximum. Default is `1`.

- `attn_factor` (int, optional): A factor affecting the positional embeddings for attention. Default is `1`.

- `beta_fast` (int, optional): A parameter used for interpolation. Default is `32`.

- `beta_slow` (int, optional): A parameter used for interpolation. Default is `1`.

- `finetuned` (bool, optional): Whether to use finetuned embeddings. Default is `False`.

- `device` (torch.device, optional): If specified, the device to which tensors will be moved.

### Forward Method <a name="forward-method"></a>

The `forward` method of the `YarnEmbedding` class applies advanced positional embeddings to the input tensor. It can be called as follows:

```python
output = yarn_embedding(input_tensor, seq_len)
```

- `input_tensor` (Tensor): The input tensor to which positional embeddings will be applied.

- `seq_len` (int): The length of the sequence for which embeddings should be generated.

---

## 4. Helpers and Functions <a name="helpers-and-functions"></a>

In addition to the `YarnEmbedding` class, there are several functions provided for working with positional embeddings.

### `find_correction_dim` <a name="find-correction-dim"></a>

This function calculates the correction dimension based on the number of rotations and other parameters.

```python
correction_dim = find_correction_dim(num_rotations, dim, base, max_position_embeddings)
```

- `num_rotations` (int): The number of rotations.

- `dim` (int): The dimensionality of the positional embeddings.

- `base` (int): The base value for calculating the positional embeddings.

- `max_position_embeddings` (int): The maximum number of position embeddings.

### `find_correction_range` <a name="find-correction-range"></a>

This function calculates the correction range based on low and high rotation values.

```python
low, high = find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings)
```

- `low_rot` (int): The low rotation value.

- `high_rot` (int): The high rotation value.

- `dim` (int): The dimensionality of the positional embeddings.

- `base` (int): The base value for calculating the positional embeddings.

- `max_position_embeddings` (int): The maximum number of position embeddings.

### `linear_ramp_mask` <a name="linear-ramp-mask"></a>

This function generates a linear ramp mask.

```python
ramp_mask = linear_ramp_mask(min, max, dim)
```

- `min` (float): The minimum value.

- `max` (float): The maximum value.

- `dim` (int): The dimensionality of the mask.

### `get_mscale` <a name="get-mscale"></a>

This function calculates the scale factor for positional embeddings.

```python
mscale = get_mscale(scale)
```

- `scale` (float): The scale factor.

---

## 5. Usage Examples <a name="usage-examples"></a>

Let's explore some usage examples of the `YarnEmbedding` class and related functions to understand how to use them effectively.

### Using the `YarnEmbedding` Class <a name="using-the-yarnembedding-class"></a>

```python
import torch

from zeta.nn import YarnEmbedding

# Create an instance of YarnEmbedding
yarn_embedding = YarnEmbedding(dim=256, max_position_embeddings=2048)

# Apply positional embeddings to an input tensor
input_tensor = torch.rand(16, 32, 256)  # Example input tensor
output = yarn_embedding(input_tensor, seq_len=32)
```

### Using the Helper Functions <a name="using-the-helper-functions"></a>

```python
from zeta.nn import find_correction_dim, find_correction_range, linear_ramp_mask, get_mscale
import torch

# Calculate correction dimension
correction_dim = find_correction_dim(num_rotations=8, dim=256, base=10000, max_position_embeddings=2048)

# Calculate correction range
low

, high = find_correction_range(low_rot=16, high_rot=32, dim=256, base=10000, max_position_embeddings=2048)

# Generate linear ramp mask
ramp_mask = linear_ramp_mask(min=0.2, max=0.8, dim=128)

# Calculate mscale
mscale = get_mscale(scale=2.0)
```

---

## 6. Additional Information <a name="additional-information"></a>

### Positional Embeddings in Transformers <a name="positional-embeddings-in-transformers"></a>

Positional embeddings play a crucial role in transformer architectures, allowing models to capture the sequential order of data. These embeddings are especially important for tasks involving sequences, such as natural language processing (NLP) and time series analysis.

---

## 7. References <a name="references"></a>

For further information on positional embeddings and transformers, you can refer to the following resources:

- [Attention Is All You Need (Transformer)](https://arxiv.org/abs/1706.03762) - The original transformer paper.

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Official PyTorch documentation for related concepts and functions.

This documentation provides a comprehensive overview of the Zeta library's `YarnEmbedding` class and related functions. It aims to help you understand the purpose, functionality, and usage of these components for advanced positional embeddings in your deep learning projects.