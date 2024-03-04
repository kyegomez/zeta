# LocalMHA: Local Multi-Head Attention for PyTorch

## Overview

The `LocalMHA` module is a local multi-head attention mechanism designed to process sequences in smaller, fixed-size windows, allowing it to handle long sequences more efficiently. This module is especially useful when working with long sequences where global attention mechanisms become computationally expensive. It combines local attention with the power of multi-head attention to capture information from different representation subspaces.

Key Concepts:

- **Local Attention**: Instead of attending to all positions in the input sequence, local attention restricts the attention to a small fixed-sized window around each position.

- **Multi-Head Attention**: The input is split into multiple heads, allowing the network to attend to information from different representation subspaces simultaneously.

## Class Definition

```python
class LocalMHA(nn.Module):
```

### Parameters:

- `dim (int)`: Dimensionality of the input sequence.
  
- `window_size (int)`: The size of the local attention window. The module will attend to this fixed-size window around each position.
  
- `dim_head (int, optional)`: Dimensionality of each attention head. Default is 64.
  
- `heads (int, optional)`: Number of attention heads. Default is 8.
  
- `dropout (float, optional)`: Dropout probability applied after the attention mechanism. Default is 0.1.
  
- `causal (bool, optional)`: If set to `True`, the attention mechanism will be causal, ensuring that each position only attends to previous positions. Default is `False`.
  
- `prenorm (bool, optional)`: If set to `True`, layer normalization is applied before the multi-head attention mechanism. Default is `False`.
  
- `qk_rmsnorm (bool, optional)`: If set to `True`, root mean square normalization is applied to the query and key tensors. Default is `False`.

- `qk_scale (int, optional)`: Scaling factor for queries and keys when `qk_rmsnorm` is set to `True`. Default is 8.

- `use_xpos (bool, optional)`: If set to `True`, the attention mechanism uses relative positional embeddings. Default is `False`.

- `xpos_scale_base (float, optional)`: Base scaling factor for relative positional embeddings. If `None`, it defaults to the square root of the dimension of the model. Only used when `use_xpos` is `True`.

- `exact_windowsize (bool, optional)`: If set to `True`, the attention window size is strictly adhered to, without any additional padding. Default is `True`.

### Method: `forward`

This method performs the forward pass of the `LocalMHA` module.

#### Parameters:

- `x (torch.Tensor)`: The input tensor with shape `[batch_size, sequence_length, dim]`.

- `mask (torch.Tensor, optional)`: A boolean mask tensor with shape `[batch_size, sequence_length]`. Positions with `True` values will be masked and won't be attended to.

- `attn_bias (torch.Tensor, optional)`: Additional bias to add to the attention scores before softmax. 

#### Returns:

- `torch.Tensor`: The output tensor after local multi-head attention with shape `[batch_size, sequence_length, dim]`.

## Example Usage

```python
from torch import tensor

from zeta import LocalMHA

# Sample data
x = tensor(
    [[...], [...], ...]
)  # Example input tensor with shape [batch_size, sequence_length, dim]

# Initialize the LocalMHA module
local_mha = LocalMHA(dim=512, window_size=5)

# Forward pass
output = local_mha(x)
```

## Mathematical Formula

For a given input \( x \):

1. Linearly project \( x \) into queries \( Q \), keys \( K \), and values \( V \).
2. If `qk_rmsnorm` is `True`, apply RMS normalization to \( Q \) and \( K \).
3. For each position \( i \) in \( x \), compute attention scores with all positions in the window around \( i \).
4. Apply softmax to the scores, then compute the attention output as a weighted sum of \( V \) based on these scores.
5. Finally, concatenate all head outputs and linearly project to get the final output.

## Additional Information

The `LocalMHA` module provides a balance between computational efficiency and the ability to capture long-range dependencies. While it restricts attention to local windows, the use of multi-head attention allows it to attend to different features within that window. The optional use of RMS normalization and relative positional embeddings further extends its capabilities.

## References

For a deeper understanding of multi-head attention, see the original [Transformer paper](https://arxiv.org/abs/1706.03762). For details on local attention, you might refer to relevant literature on efficient transformers or localized attention mechanisms.