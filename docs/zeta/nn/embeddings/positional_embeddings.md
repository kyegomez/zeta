# Zeta Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose and Functionality](#purpose-and-functionality)
3. [Class: `PositionalEmbedding`](#class-positionalembedding)
   - [Initialization](#initialization)
   - [Parameters](#parameters)
   - [Forward Method](#forward-method)
4. [Usage Examples](#usage-examples)
   - [Basic Usage](#basic-usage)
   - [Customized Positional Embeddings](#customized-positional-embeddings)
   - [Using Provided Positions](#using-provided-positions)
5. [Additional Information](#additional-information)
   - [Positional Embeddings in Transformers](#positional-embeddings-in-transformers)
6. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the Zeta documentation for the `PositionalEmbedding` class! Zeta is a powerful library for deep learning in PyTorch, and this documentation will provide a comprehensive understanding of the `PositionalEmbedding` class. 

---

## 2. Purpose and Functionality <a name="purpose-and-functionality"></a>

The `PositionalEmbedding` class is a key component in sequence modeling tasks, particularly in transformers. It is designed to create positional embeddings that provide essential information about the position of tokens in a sequence. Below, we will explore its purpose and functionality.

---

## 3. Class: `PositionalEmbedding` <a name="class-positionalembedding"></a>

The `PositionalEmbedding` class is used to generate positional embeddings for sequences. These embeddings are vital for transformers and other sequence-based models to understand the order of elements within the input data.

### Initialization <a name="initialization"></a>

To create a `PositionalEmbedding` instance, you need to specify various parameters. Here's an example of how to initialize it:

```python
PositionalEmbedding(
    num_embeddings,
    embedding_dim,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
)
```

### Parameters <a name="parameters"></a>

- `num_embeddings` (int): The total number of embeddings to generate. This typically corresponds to the sequence length.

- `embedding_dim` (int): The dimensionality of the positional embeddings. This should match the dimensionality of the input data.

- `padding_idx` (int, optional): If specified, the embeddings at this position will be set to all zeros. Default is `None`.

- `max_norm` (float, optional): If specified, the embeddings will be normalized to have a maximum norm of this value. Default is `None`.

- `norm_type` (float, optional): The type of norm to apply if `max_norm` is specified. Default is `2.0`.

- `scale_grad_by_freq` (bool, optional): If `True`, the gradients of the embeddings will be scaled by the frequency of the corresponding positions. Default is `False`.

- `sparse` (bool, optional): If `True`, a sparse tensor will be used for embeddings. Default is `False`.

### Forward Method <a name="forward-method"></a>

The `forward` method of `PositionalEmbedding` generates positional embeddings based on the input positions. It can be called as follows:

```python
output = positional_embedding(positions)
```

- `positions` (Tensor): A tensor containing the positions for which you want to generate positional embeddings.

---

## 4. Usage Examples <a name="usage-examples"></a>

Let's explore some usage examples of the `PositionalEmbedding` class to understand how to use it effectively.

### Basic Usage <a name="basic-usage"></a>

```python
import torch

from zeta.nn import PositionalEmbedding

# Create a PositionalEmbedding instance
positional_embedding = PositionalEmbedding(num_embeddings=100, embedding_dim=128)

# Generate positional embeddings for a sequence of length 10
positions = torch.arange(10)
embeddings = positional_embedding(positions)
```

### Customized Positional Embeddings <a name="customized-positional-embeddings"></a>

You can customize the positional embeddings by specifying additional parameters such as `max_norm` and `scale_grad_by_freq`.

```python
import torch

from zeta.nn import PositionalEmbedding

# Create a PositionalEmbedding instance with customization
positional_embedding = PositionalEmbedding(
    num_embeddings=100, embedding_dim=128, max_norm=1.0, scale_grad_by_freq=True
)

# Generate positional embeddings for a sequence of length 10
positions = torch.arange(10)
embeddings = positional_embedding(positions)
```

### Using Provided Positions <a name="using-provided-positions"></a>

You can also provide your own positions when generating positional embeddings.

```python
import torch

from zeta.nn import PositionalEmbedding

# Create a PositionalEmbedding instance
positional_embedding = PositionalEmbedding(num_embeddings=100, embedding_dim=128)

# Provide custom positions for embedding
custom_positions = torch.tensor([5, 10, 15, 20])
embeddings = positional_embedding(custom_positions)
```

---

## 5. Additional Information <a name="additional-information"></a>

### Positional Embeddings in Transformers <a name="positional-embeddings-in-transformers"></a>

Positional embeddings play a crucial role in transformers and other sequence-to-sequence models. They allow the model to understand the order of elements in a sequence, which is essential for tasks like language translation and text generation.

---

## 6. References <a name="references"></a>

- [Attention Is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)

This documentation provides a comprehensive guide to the `PositionalEmbedding` class in the Zeta library, explaining its purpose, functionality, parameters, and usage. You can now effectively integrate this class into your deep learning models for various sequence-based tasks.