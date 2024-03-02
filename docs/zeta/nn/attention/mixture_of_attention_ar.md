# MixtureOfAutoregressiveAttention

The `MixtureOfAutoregressiveAttention` module provides a versatile attention mechanism that combines the strength of local multi-head attention with a flexible routing mechanism, enabling attention to focus on specific parts of the input sequence depending on the context. The mixture strategy optimizes the computation while retaining the benefits of the global attention perspective.

## Overview:
This module integrates local multi-head attention with coordinate descent-based routing, which chooses certain query and key-value pairs for further processing, allowing the model to focus on specific parts of the sequence that are more relevant for the task at hand. 

## Key Concepts:

- **Local Attention**: A mechanism where each position can attend to a restricted window of positions around it.
- **Routing**: Mechanism to select specific tokens based on importance for specialized processing.
- **Coordinate Descent Routing**: A strategy to decide which tokens should be routed for specialized processing.
- **Null Tokens**: Default tokens used for positions that weren't routed to any expert.

## Class Definition:

```python
class MixtureOfAutoregressiveAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_routed_queries: int,
        num_routed_key_values: int,
        local_attn_window_size: int,
        routed_window_size: Optional[int] = None,
        num_experts: int = 2,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        use_triton: bool = False,
        flash_attn: bool = True,
        prenorm: bool = True,
        average_routed: bool = False,
        **kwargs,
    ):
        ...
```

### Parameters:

- `dim` (int): Dimensionality of the input sequence.
- `num_routed_queries` (int): Number of queries to be routed.
- `num_routed_key_values` (int): Number of key-values to be routed.
- `local_attn_window_size` (int): Window size for local attention.
- `routed_window_size` (int, optional): Window size for routing. Defaults to `local_attn_window_size`.
- `num_experts` (int, optional): Number of experts. Defaults to 2.
- `dim_head` (int, optional): Dimensionality of each attention head. Defaults to 64.
- `heads` (int, optional): Number of attention heads. Defaults to 8.
- `dropout` (float, optional): Dropout probability. Defaults to 0.
- `use_triton` (bool, optional): Flag to use Triton for optimization. Defaults to False.
- `flash_attn` (bool, optional): Flag to use flash attention mechanism. Defaults to True.
- `prenorm` (bool, optional): Flag for pre-normalization. Defaults to True.
- `average_routed` (bool, optional): Whether to average the routed tokens or not. Defaults to False.

### Methods:

#### forward

```python
def forward(
    self,
    x: torch.Tensor,
    rotary_emb: Optional[torch.Tensor] = None,
    num_routed_queries: Optional[int] = None,
    num_routed_key_values: Optional[int] = None,
) -> torch.Tensor:
    ...
```

- `x` (torch.Tensor): Input tensor of shape `(batch_size, sequence_length, dim)`.
- `rotary_emb` (torch.Tensor, optional): Rotary embeddings. Defaults to None.
- `num_routed_queries` (int, optional): Overrides the number of queries to be routed. Defaults to class defined value.
- `num_routed_key_values` (int, optional): Overrides the number of key-values to be routed. Defaults to class defined value.

## Examples:

1. **Basic Usage**

```python
from zeta.nn import MixtureOfAutoregressiveAttention

attention_layer = MixtureOfAutoregressiveAttention(
    dim=512, num_routed_queries=5, num_routed_key_values=5, local_attn_window_size=32
)
x = torch.randn(10, 60, 512)
out = attention_layer(x)
```

2. **With Rotary Embeddings**

```python
rotary_emb = torch.randn(60, 512)
out = attention_layer(x, rotary_emb=rotary_emb)
```

3. **Changing Routing Parameters**

```python
out = attention_layer(x, num_routed_queries=3, num_routed_key_values=7)
```

## Mathematical Description:

Let \(x\) be the input sequence and \(w\) be the attention window size. The local attention output \(L\) for \(x\) is computed based on a restricted window of positions around each position in \(x\). 

The queries and key-values are then routed based on their importance scores to produce the routed attention output \(R\). 

The final output \(O\) is a combination of the local attention output \(L\) and the routed attention output \(R\):

\[ O = f(L, R) \]

where \(f\) is a combination function, which

 might be a weighted sum, concatenation, or other methods.

## Use Cases:

This module is best suited for sequence-to-sequence models where a mix of local and global attention is required. It could be advantageous in applications like:

- Long document summarization
- Language modeling
- Machine translation

## Conclusion:

The `MixtureOfAutoregressiveAttention` module provides a combination of local attention and flexible routing, allowing models to focus on specific parts of an input sequence that are contextually relevant. This can lead to more efficient computation and potentially better performance on sequence processing tasks.