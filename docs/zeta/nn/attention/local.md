# `LocalAttention` Module Documentation

## Overview and Introduction

The `LocalAttention` module provides a mechanism to perform local attention operations. Unlike global attention where every token can attend to every other token, in local attention each token can only attend to a subset of tokens within a defined window. This reduces the computational cost and captures the local structure in sequences like text or time-series data.

Key terms:

- **Local Attention**: A type of attention mechanism where a token attends only to a subset of tokens within a specified window.
  
- **Causal Attention**: Ensures that an output token at time `t` can only attend to input tokens at times `<= t`.
  
- **Rotary Positional Embeddings**: A technique for incorporating sequence position information without the need for additional position-specific parameters.

## Class Definition

```python
class LocalAttention(nn.Module):
    ...
```

### Parameters

- `window_size`: (int) The size of the attention window.

- `causal`: (bool, optional) If set to `True`, ensures causal attention. Default: `False`.

- `look_backward`: (int, optional) How many positions to look backward from the current position. Default: `1`.

- `look_forward`: (int, optional) How many positions to look forward from the current position. Default: `None` which implies 0 if causal is `True`.

- `dropout`: (float, optional) Dropout rate for attention weights. Default: `0.1`.

- `shared_qk`: (bool, optional) If set to `True`, the query and key are the same. Useful for certain types of attention mechanisms. Default: `False`.

- `rel_pos_emb_config`: (Optional) Deprecated. Configuration for the relative positional embeddings.

- `dim`: (int, optional) Dimension of embeddings. Only needed if `rel_pos_emb_config` is not provided.

- `autopad`: (bool, optional) If set to `True`, sequence will be automatically padded to be divisible by the window size. Default: `False`.

- `exact_windowsize`: (bool, optional) Ensures exact window size for non-causal attention. Default: `False`.

- `scale`: (Optional) Scaling factor for the queries.

- `use_rotary_pos_emb`: (bool, optional) If set to `True`, rotary positional embeddings will be used. Default: `True`.

- `use_xpos`: (bool, optional) If set to `True`, allows for extrapolation of window sizes. Requires `use_rotary_pos_emb` to be `True`. Default: `False`.

- `xpos_scale_base`: (Optional) Base scaling factor for extrapolated window sizes.

### Forward Method

#### Parameters

- `q`: (Tensor) The query tensor.

- `k`: (Tensor) The key tensor.

- `v`: (Tensor) The value tensor.

- `mask`: (Optional[Tensor]) A mask tensor for the keys. Can also be passed as `input_mask`.

- `input_mask`: (Optional[Tensor]) Another way to pass the mask tensor for keys.

- `attn_bias`: (Optional[Tensor]) Additional biases to add to the attention scores.

- `window_size`: (Optional[int]) If provided, this window size will override the default window size defined during initialization.

#### Returns

- `out`: (Tensor) The output tensor after the attention operation.

## Functionality and Usage

The `LocalAttention` module is designed to efficiently compute attention values over a local window. When the `forward` method is called, the module performs the following steps:

1. Reshape and, if required, autopad the input tensors.
2. Calculate the attention scores between the queries and keys.
3. Optionally apply causal masking and other types of masking.
4. Calculate the softmax over the attention scores.
5. Use the attention scores to weight the value tensor and produce the output.

### Usage Example:

```python
import torch
import torch.nn as nn

from zeta import LocalAttention

q = torch.randn(1, 100, 32)
k = torch.randn(1, 100, 32)
v = torch.randn(1, 100, 32)

local_attn = LocalAttention(window_size=5, causal=True, dim=32)
out = local_attn(q, k, v)
```

## Additional Information and Tips

- When using `LocalAttention` with `causal=True`, ensure that `look_forward` is not set to a value greater than 0.

- The `autopad` option can be helpful when dealing with sequences of arbitrary lengths, but may introduce padding tokens.

## References and Resources

For a deeper understanding of attention mechanisms and their local variants:

- Vaswani, A. et al. (2017) "Attention Is All You Need". Advances in Neural Information Processing Systems 30.

- Liu, Z. et al. (2018) "Generating Wikipedia by Summarizing Long Sequences". International Conference on Learning Representations.

### Simple Mathematical Formula

Given a sequence of length \( n \), each token attends to tokens within a window of size \( w \) around it. The attention scores \( A \

) between query \( q \) and key \( k \) are given by:

\[ A = \text{softmax} \left( \frac{q \times k^T}{\sqrt{d}} \right) \]

Where \( d \) is the dimension of the embeddings.

## Conclusion

The `LocalAttention` module provides a computationally efficient way to apply attention mechanisms to local windows within a sequence. By using parameters such as `window_size` and `causal`, users can fine-tune the attention behavior to fit their specific needs. The module's flexible design and variety of options make it a valuable tool for many sequence modeling tasks.