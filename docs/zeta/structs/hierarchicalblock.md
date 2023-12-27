# Module/Class Name: HierarchicalBlock

## Overview

The HierarchicalBlock class in the pyTorch library is an implementation of the hierarchical token-wise attention mechanism used in some transformer models. Hierarchical token-wise attention allows a model to selectively focus on portions of the input sequence, thus the model can efficiently learn longer-range dependencies in the input data. 

It uses "nn.Module", which is a base class for all neural network modules from the PyTorch library. HierarchicalBlock provides the functionality to handle the hierarchical structure and neural network layers within the block.

It is recommended to use this class, rather than handle the hierarchical structure of a neural network manually to ensure the hierarchical structure has an ordered representation.

### Purpose 

The HierarchicalBlock class allows efficient modelling of attention in transformer models, enabling the model to learn long-range dependencies in the input data. This is especially useful for large-scale Natural Language Processing tasks like language translation and text summarization where long sequences of text need to be processed.

The design of HierarchicalBlock ensures appropriate assignment and registration of submodules, which converts the parameters appropriately when methods like :meth:`to` etc. are called. 

It has the `:ivar training` variable to represent whether the module is in training or evaluation mode.

The HierarchicalBlock class is vital for building complex models and ensuring submodules are correctly registered and parameters updated.


# HierarchicalBlock Class Definition


```python
class HierarchicalBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, window_size=None, compress_factor=1, stride=1, ff_mult=4):
    ...
```

## Class Parameters

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| dim | int | Defines the dimension of the model. |
| dim_head | int | Determines the head dimensions. Default value is 64. |
| heads | int | Determines the number of parallel attention heads. Default value is 8. |
| window_size | int or NoneType | If a value exists, it specifies the size of the window for local Multihead Attention (LocalMHA). If no value exists, a standard Attention operation will be performed. Default is None. |
| compress_factor | int | Factor by which to compress inputs. Must be a power of two. Default is 1 (no compression). |
| stride | int | Stride size for the attention operation. Default is 1. |
| ff_mult | int | Multiplier for the dimension of the feed forward network hidden layer. This is used to expand the inner hidden layer of the model from the input sequence. |


## Methods

### forward

```python
def forward(self, x):
    ...
```

## Method Parameters and returns

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| x | Tensor or array-like | The input tensor to the HierarchicalBlock instance. |

**Returns:**

| Return Variables | Type  | Description |
| ---------------- | ----  | ----------- |
| x | Tensor or array-like | Returns the tensor after it has been processed through the 'attn' (attention) and 'ff' (feed forward) operations, and optionally compressed and padded. It returns a tensor with the same batch size but with a different sequence length, depending on the size of the window used in 'attn' and the settings of 'compress_factor' and 'stride'. |

## Usage Example

Import necessary modules and define an input sequence:

```python
import torch
import torch.nn as nn
from functools import partial
from utils import is_power_of_two, pad_seq_to_multiple, token_shift, rearrange, exists

sequence_length = 10
batch_size = 32
dim = 512

x = torch.randn(batch_size, sequence_length, dim)

# Define an instance of HierarchicalBlock
hierarchical_block = HierarchicalBlock(dim=dim)

# Apply the forward method of the hierarchical_block instance to x
out = hierarchical_block.forward(x)
```
In the example above, we first import the necessary modules. We initialize a tensor `x` with random numbers, having batch_size of 32, sequence_length of 10, and dimension of 512. We define an instance of HierarchicalBlock where `dim = 512`. We then pass the tensor `x` to the forward method to get the output tensor.
