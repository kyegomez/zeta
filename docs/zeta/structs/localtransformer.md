# LocalTransformer

## Introduction

The `LocalTransformer` is a powerful machine learning module that implements a sequence-to-sequence model based on the local self-attention module part of the Transformer architecture. This module is specifically designed for applications where sequences of tokens are transformed, such as natural language processing tasks. 

At a high level, a transformer takes in a sequence of tokens and outputs a new sequence of tokens. Local transformer creates a module where attention is based on a limited window of the input sequence which can be beneficial for both efficiency and model performance in certain cases.

## Definitions and Key Concepts

- **tokens**: Individual elements of a sequence, typically words in a sentence for language tasks.
- **sequence length**: The number of tokens in each sequence.
- **embeddings**: Vector representations of tokens, which allow them to be processed by the network.
- **attention**: A mechanism in transformers that allows the model to focus on different parts of the input when producing each part of the output. 

## Class Definition

The class signature for the `LocalTransformer` is as follows:

```
class LocalTransformer(nn.Module):
```

## Arguments

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| num_tokens | int | The number of tokens in the input vocabulary. | - |
| max_seq_len | int | The maximum sequence length. | - |
| dim | int | The dimensionality of the token and positional embeddings. | - |
| depth | int | The number of transformer layers. | - |
| causal | bool | Whether to use causal attention or not. | True |
| local_attn_window_size | int | The size of the local attention window. | 512 |
| dim_head | int | The dimensionality of each attention head. | 64 |
| heads | int | The number of attention heads. | 8 |
| ff_mult | int | The multiplier for the feedforward network dimension. | 4 |
| attn_dropout | float | The dropout rate for attention layers. | 0.0 |
| ff_dropout | float | The dropout rate for feedforward layers. | 0.0 |
| ignore_index | int | The index to ignore during loss calculation. | -1 |
| use_xpos | bool | Whether to use positional embeddings based on xpos. | False |
| xpos_scale_base | None | The base value for scaling xpos positional embeddings. | None |
| use_dynamic_pos_bias | bool | Whether to use dynamic positional bias or not. | False |


### Understanding Arguments
    
- **num_tokens**: This determines the size of the vocabulary. This is set according to the dataset and cannot be modified post initialization.
- **max_seq_len**: This sets the maximum sequence length. As the model would need to create key, query and values for each token, increasing this value can lead to a significant increase in memory usage.
- **dim**: This is the size of the model's embeddings. The higher this value, the more information each embedding can store. However, similarly to max_seq_len, this can also drastically increase memory usage. 
- **depth**: This corresponds to the number of layers the model will have. Deeper models can potentially have better representative power, but it can also lead to overfitting and longer training times.

## Attributes

| Attribute | Description |
| --- | --- |
| token_emb | Embedding layer for token embeddings. |
| pos_emb | Embedding layer for positional embeddings. |
| max_seq_len | The maximum sequence length. |
| layers | List of transformer layers. |
| local_attn_window_size | The size of the local attention window. |
| dynamic_pos_bias | Dynamic positional bias layer, if enabled. |
| ignore_index | The index to ignore during loss calculation. |
| to_logits | Sequential layer for converting transformer output to logits. |

## Example

The following example demonstrates how to initialize and use the `LocalTransformer` class for a simple task:

```python
import torch
from zeta.structs import LocalTransformer

# Define a LocalTransformer
model = LocalTransformer(num_tokens=500, max_seq_len=10, dim=32, depth=2)

# Define a simple sequence
sequence = torch.randint(0, 500, (1, 10))

# Forward pass
output = model(sequence)

```

This will create a `LocalTransformer` model with a vocabulary of size 500, a maximum sequence length of 10, an embedding dimension of 32, and 2 transformer layers. It then performs a forward pass of the sequence through the model, outputting the transformed sequence.

## Conclusion

The `LocalTransformer` module is a highly flexible and modular implementation of the transformer architecture, equipped with local attention. Given its configurable nature, it is amenable to various NLP and sequence-to-sequence modeling tasks. An understanding of its input arguments, attributes, and overall design is essential to leverage its full potential. 

For any additional details or queries, please refer to external resources or related papers for an in-depth understanding of Transformers in Machine Learning.
