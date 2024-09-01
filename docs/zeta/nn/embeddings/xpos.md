# `XPOS` Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose and Functionality](#purpose-and-functionality)
3. [Class: `XPOS`](#class-xpos)
   - [Initialization](#initialization)
   - [Parameters](#parameters)
   - [Forward Method](#forward-method)
4. [Functions](#functions)
   - [`fixed_pos_embedding`](#fixed-pos-embedding)
   - [`rotate_every_two`](#rotate-every-two)
   - [`duplicate_interleave`](#duplicate-interleave)
   - [`apply_rotary_pos_emb`](#apply-rotary-pos-emb)
5. [Usage Examples](#usage-examples)
   - [Using the `XPOS` Class](#using-the-xpos-class)
   - [Using the Functions](#using-the-functions)
6. [Additional Information](#additional-information)
   - [Positional Embeddings in Transformers](#positional-embeddings-in-transformers)
7. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the Zeta documentation for the `XPOS` class and related functions! Zeta is a powerful library for deep learning in PyTorch, and this documentation will provide a comprehensive understanding of the `XPOS` class and its associated functions.

---

## 2. Purpose and Functionality <a name="purpose-and-functionality"></a>

The `XPOS` class and its related functions are designed to generate and apply rotary positional embeddings to input tensors. These embeddings are crucial for sequence-to-sequence models, particularly in transformer architectures. Below, we will explore their purpose and functionality.

---

## 3. Class: `XPOS` <a name="class-xpos"></a>

The `XPOS` class is used to apply rotary positional embeddings to input tensors. These embeddings are essential for transformers to understand the positional information of elements in a sequence.

### Initialization <a name="initialization"></a>

To create an instance of the `XPOS` class, you need to specify the following parameters:

```python
XPOS(
    head_dim: int = None,
    scale_base: int = 512
)
```

### Parameters <a name="parameters"></a>

- `head_dim` (int, optional): The dimensionality of the positional embeddings. If not specified, it defaults to `None`, which is used to calculate the dimension based on the input tensor. It is recommended to set this value explicitly for consistency.

- `scale_base` (int, optional): The base value for scaling the positional embeddings. Default is `512`.

### Forward Method <a name="forward-method"></a>

The `forward` method of the `XPOS` class applies rotary positional embeddings to the input tensor. It can be called as follows:

```python
output = xpos(input_tensor, offset=0, downscale=False)
```

- `input_tensor` (Tensor): The input tensor to which positional embeddings will be applied.

- `offset` (int, optional): An offset value for positional embeddings. Default is `0`.

- `downscale` (bool, optional): If `True`, the positional embeddings are downscaled. Default is `False`.

---

## 4. Functions <a name="functions"></a>

In addition to the `XPOS` class, there are several functions provided for working with positional embeddings.

### `fixed_pos_embedding` <a name="fixed-pos-embedding"></a>

This function generates fixed sine and cosine positional embeddings based on the input tensor's scale.

```python
sin, cos = fixed_pos_embedding(x)
```

- `x` (Tensor): Input tensor of shape `(seq_len, dim)`.

### `rotate_every_two` <a name="rotate-every-two"></a>

This function rearranges the elements of the input tensor by rotating every two elements.

```python
output_tensor = rotate_every_two(input_tensor)
```

- `input_tensor` (Tensor): Input tensor of shape `(batch_size, seq_len, dim)`.

### `duplicate_interleave` <a name="duplicate-interleave"></a>

This function duplicates a matrix while interleaving the copy.

```python
duplicated_matrix = duplicate_interleave(matrix)
```

- `matrix` (Tensor): Input matrix.

### `apply_rotary_pos_emb` <a name="apply-rotary-pos-emb"></a>

This function applies rotary positional embeddings to the input tensor.

```python
output_tensor = apply_rotary_pos_emb(input_tensor, sin, cos, scale=1)
```

- `input_tensor` (Tensor): Input tensor of shape `(batch_size, seq_len, dim)`.
- `sin` (Tensor): Sine positional embeddings of shape `(seq_len, dim)`.
- `cos` (Tensor): Cosine positional embeddings of shape `(seq_len, dim)`.
- `scale` (float): Scaling factor for the positional embeddings.

---

## 5. Usage Examples <a name="usage-examples"></a>

Let's explore some usage examples of the `XPOS` class and related functions to understand how to use them effectively.

### Using the `XPOS` Class <a name="using-the-xpos-class"></a>

```python
import torch

from zeta.nn import XPOS

# Create an XPOS instance
xpos = XPOS(head_dim=256, scale_base=512)

# Apply positional embeddings to an input tensor
input_tensor = torch.rand(16, 32, 256)  # Example input tensor
output = xpos(input_tensor, offset=0, downscale=False)
```

### Using the Functions <a name="using-the-functions"></a>

```python
import torch

from zeta.nn import (
    apply_rotary_pos_emb,
    duplicate_interleave,
    fixed_pos_embedding,
    rotate_every_two,
)

# Generate fixed positional embeddings
input_tensor = torch.rand(32, 512)  # Example input tensor
sin, cos = fixed_pos_embedding(input_tensor)

# Rotate every two elements in a tensor
input_tensor = torch.rand(16, 64, 256)  # Example input tensor
output_tensor = rotate_every_two(input_tensor)

# Duplicate and interleave a matrix
input_matrix = torch.rand(8, 8)  # Example input matrix
duplicated_matrix = duplicate_interleave(input_matrix)

# Apply rotary positional embeddings
input_tensor = torch.rand(16, 32, 256)  # Example input tensor
output_tensor = apply_rotary_pos_emb(input_tensor, sin, cos, scale=1)
```

---

## 6. Additional Information <a name="additional-information"></a>

### Positional Embeddings in Transformers <a name="positional-embeddings-in-transformers"></a>

Positional embeddings play a crucial role in transformers and other sequence-to-sequence models. They enable the model to understand the order of elements in a sequence, which is essential for tasks like natural language processing, machine translation, and text generation.

---

## 7. References <a name="references"></a>

This documentation provides a comprehensive guide to the `XPOS` class and related functions in the Zeta library, explaining their purpose, functionality, parameters, and usage. You can now effectively integrate these components into your deep learning models, particularly in transformer-based architectures, for various sequence-based tasks.

For further information on the underlying concepts and principles of positional embeddings in

 transformers, you may refer to the original paper:

- [Attention Is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)

Please consult the official PyTorch documentation for any specific PyTorch-related details: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html).