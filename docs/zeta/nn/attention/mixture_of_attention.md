## `MixtureOfAttention` Class in the Zeta Library

---

### Overview

`MixtureOfAttention` is a powerful and versatile attention mechanism in the Zeta library. It uniquely combines the ideas of dynamic routing and local attention. The class enables the model to focus on different portions of the input data by creating multiple routes for queries and key-values. This makes it particularly effective for tasks that require flexible attention over the input data.

---

### Class Definition

#### MixtureOfAttention

```python
class MixtureOfAttention(nn.Module):
```

##### Parameters:

- **dim (int)**: The dimension of the input tensor.
  
- **num_routed_queries (int)**: Number of routed queries.

- **num_routed_key_values (int)**: Number of routed key-value pairs.

- **dim_context (int, optional)**: The dimension of the context tensor. Defaults to the value of `dim`.

- **local_attn (bool, optional)**: Whether to use local attention. Defaults to False.

- **local_attn_window_size (int, optional)**: The window size for local attention if `local_attn` is set to True.

- **num_experts (int)**: Number of expert routes.

- **dim_head (int, optional)**: Dimension of each attention head. Defaults to 64.

- **heads (int, optional)**: Number of attention heads. Defaults to 8.

- **dropout (float, optional)**: Dropout probability. Defaults to 0.1.

- **use_triton (bool, optional)**: Whether to use Triton for optimized computation. Defaults to True.

- **flash_attn (bool, optional)**: Whether to use flash attention mechanism. Defaults to True.

- **prenorm (bool, optional)**: Whether to use pre-normalization in attention. Defaults to True.

- **average_routed (bool, optional)**: Whether to average the routed queries and key-values. Defaults to False.

- **kwargs**: Additional keyword arguments.

---

### Functionality and Usage

`MixtureOfAttention` offers the ability to combine different attention mechanisms, enabling it to better adapt to the task at hand. Its core functionality hinges on the routing mechanism, which dynamically determines which parts of the input should be focused on. When combined with local attention, this mechanism allows the model to concentrate on both local and global features in the data.

#### Usage Examples:

**1. Basic usage with default parameters:**

```python
import torch

from zeta.nn import MixtureOfAttention

dim = 512
model = MixtureOfAttention(
    dim, num_routed_queries=100, num_routed_key_values=100, num_experts=4
)
x = torch.rand(16, 50, dim)
output = model(x)
```

**2. Using local attention:**

```python
import torch

from zeta.nn import MixtureOfAttention

dim = 512
model = MixtureOfAttention(
    dim,
    num_routed_queries=100,
    num_routed_key_values=100,
    num_experts=4,
    local_attn=True,
    local_attn_window_size=5,
)
x = torch.rand(16, 50, dim)
output = model(x)
```

**3. Using pre-normalization and dropout:**

```python
import torch

from zeta.nn import MixtureOfAttention

dim = 512
model = MixtureOfAttention(
    dim,
    num_routed_queries=100,
    num_routed_key_values=100,
    num_experts=4,
    prenorm=True,
    dropout=0.1,
)
x = torch.rand(16, 50, dim)
output = model(x)
```

---

### Mathematical Formulation

Given an input tensor \( x \) of shape \( (batch\_size, seq\_len, dim) \), the model first determines which tokens should be routed using the `query_router` and `key_value_router`. These routed tokens act as "experts" and are processed using the attention mechanism.

If using local attention, for each token in \( x \), a local window of size `local_attn_window_size` is considered around it.

For the routed tokens (either global or local), the attention scores are computed using:

\[ \text{Attention}(Q, K) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V \]

Where \( Q \), \( K \), and \( V \) are the query, key, and value matrices, and \( d_k \) is the dimension of the keys.

The final output is a combination of the attention outputs from these different mechanisms, based on the configuration.

---

### Additional Information and Tips

- If both local attention and global attention are enabled, make sure to provide a valid `local_attn_window_size`.
  
- Using `use_triton=True` can optimize the computation using the Triton framework, but ensure you have Triton support in your environment.
  
- The `flash_attn` mechanism can further enhance attention computation speed.

---

### References and Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper which introduced the multi-head attention mechanism.

- [Local Attention](https://arxiv.org/abs/2004.13621) - A paper discussing the benefits of local attention.

- [Triton](https://triton-lang.org/) - An open-source domain-specific language to help researchers write fast GPU code. 

For more details and advanced usage scenarios, please refer to the official Zeta library documentation.