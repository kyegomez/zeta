# `MultiModalCrossAttention` Documentation

## Overview

The `MultiModalCrossAttention` module is an enhanced cross-attention mechanism designed for various multimodal tasks, such as combining information from different sources (e.g., text and images) in a transformer-based architecture. This module extends the standard self-attention mechanism by providing features like conditional layer normalization, lambda masking, and dropout for improved modeling of multimodal data.

This documentation provides a comprehensive guide to the `MultiModalCrossAttention` module, explaining its architecture, purpose, parameters, and usage through detailed examples.

## Table of Contents

1. [Module Overview](#module-overview)
2. [Installation](#installation)
3. [Module Architecture](#module-architecture)
4. [Parameters](#parameters)
5. [Usage Examples](#usage-examples)
   - [Example 1: Basic Usage](#example-1-basic-usage)
   - [Example 2: Conditional Layer Normalization](#example-2-conditional-layer-normalization)
   - [Example 3: Lambda Masking](#example-3-lambda-masking)
6. [Additional Information and Tips](#additional-information-and-tips)

## Installation

Before using the `MultiModalCrossAttention` module, you need to ensure that you have the required dependencies installed. Here are the dependencies:

- PyTorch
- Einops
- TorchVision (for the examples)

You can install these dependencies using `pip`:

```bash
pip install zetascale
```

Now let's delve into the architecture, parameters, and usage of the `MultiModalCrossAttention` module.

## Module Architecture

The `MultiModalCrossAttention` module extends the standard self-attention mechanism used in transformer architectures. It takes as input a query tensor `x` and a context tensor `context`, which represent the input data from different modalities. The module performs multi-head attention between these tensors, combining information from both modalities.

The key features of the `MultiModalCrossAttention` module include:

- Multi-Head Attention: The attention mechanism is split into multiple heads, allowing the model to attend to different parts of the input data in parallel.

- Conditional Layer Normalization: Optional conditional layer normalization can be applied to the query and key tensors before attention computation.

- Lambda Masking: An optional mask can be applied to the attention weights to control which elements are attended to during computation.

- Dropout: Dropout is applied to the attention weights to prevent overfitting.

- Output Projection: The module projects the attention outputs to the desired output dimension.

- Attention Strategy: The module supports two attention strategies: "average" (average attention outputs from all heads) and "concatenate" (concatenate attention outputs from all heads).

The architecture of the `MultiModalCrossAttention` module is designed to handle multimodal data efficiently by combining information from different sources. Now, let's explore the parameters of this module.

## Parameters

The `MultiModalCrossAttention` module accepts several parameters, each of which controls different aspects of its behavior. Here are the parameters:

| Parameter              | Description                                               | Default Value   |
|------------------------|-----------------------------------------------------------|-----------------|
| `dim`                  | Dimension of the model.                                   | None (Required) |
| `heads`                | Number of attention heads.                                | None (Required) |
| `context_dim`          | Dimension of the context.                                 | None (Required) |
| `dim_head`             | Dimension of each attention head.                          | 64              |
| `dropout`              | Dropout rate applied to attention weights.                | 0.1             |
| `qk`                   | Whether to use conditional layer normalization.           | False           |
| `post_attn_norm`       | Whether to use post-attention normalization.              | False           |
| `attention_strategy`   | Attention strategy: "average" or "concatenate".           | None (Required) |
| `mask`                 | Mask for lambda masking.                                   | None            |

Now that we understand the parameters, let's explore how to use the `MultiModalCrossAttention` module with detailed usage examples.

## Usage Examples

### Example 1: Basic Usage

In this example, we'll demonstrate the basic usage of the `MultiModalCrossAttention` module. We'll create an instance of the module, feed it with query and context tensors, and obtain the attention outputs.

```python
import torch
from einops import rearrange
from torch import nn

from zeta.nn import MultiModalCrossAttention

# Create a MultiModalCrossAttention module
dim = 1024
heads = 8
context_dim = 1024
attn = MultiModalCrossAttention(dim, heads, context_dim)

# Generate random query and context tensors
query = torch.randn(1, 32, dim)
context = torch.randn(1, 32, context_dim)

# Perform multi-head cross-attention
output = attn(query, context)

# Print the shape of the output
print(output.shape)
```

Output:
```
torch.Size([1, 32, 1024])
```

In this basic usage example, we create an instance of the `MultiModalCrossAttention` module and apply it to random query and context tensors, resulting in an output tensor.

### Example 2: Conditional Layer Normalization

In this example, we'll enable conditional layer normalization and observe the effect on the attention outputs.

```python
# Create a MultiModalCrossAttention module with conditional layer normalization
attn = MultiModalCrossAttention(dim, heads, context_dim, qk=True)

# Generate random query and context tensors
query = torch.randn(1, 32, dim)
context = torch.randn(1, 32, context_dim)

# Perform multi-head cross-attention
output = attn(query, context)

# Print the shape of the output
print(output.shape)
```

Output:
```
torch.Size([1, 32, 1024])
```

In this example, we enable conditional layer normalization (`qk=True`) and observe the effect on the attention outputs.

### Example 3: Lambda Masking

Lambda masking allows us to control which elements are attended to during computation. In this example, we'll apply a mask and observe how it affects the attention weights.

```python
# Create a MultiModalCrossAttention module with lambda masking
mask = torch.randint(0, 2, (32, 32), dtype=torch.bool)
attn = MultiModalCrossAttention(dim, heads, context_dim, mask=mask)

# Generate random query and context tensors
query = torch.randn(1, 32, dim)
context = torch.randn(1, 32, context_dim)

# Perform multi-head cross-attention
output = attn(query, context)

# Print the shape of the output
print(output.shape)
```

Output:
```
torch.Size([1, 32, 1024])
```

In this example, we apply a lambda mask to control attention weights and observe its effect on the attention outputs.

## Additional Information and Tips

- The `MultiModalCrossAttention` module can be integrated into various multimodal architectures to capture dependencies between different data sources effectively.

- Experiment with different values of `heads` and `dim_head` to find the optimal configuration for your task.

- You can choose the appropriate attention strategy (`average` or `concatenate`) based on your specific requirements.

- If you encounter any issues or have questions, refer to the PyTorch documentation or seek assistance from the community.

By following these guidelines and examples, you can effectively utilize the `MultiModalCrossAttention` module in your multimodal deep learning projects.