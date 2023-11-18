[![Multi-Modality](images/agorabanner.png)](https://discord.gg/qUtxnK2NMf)

![Zeta banner](images/zeta.png)
Build High-performance, agile, and scalable AI models with modular and re-useable building blocks!


[![Docs](https://readthedocs.org/projects/zeta/badge/)](https://zeta.readthedocs.io)

<p>
  <a href="https://github.com/kyegomez/zeta/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://pypi.org/project/zetascale"><img alt="MIT License" src="https://badge.fury.io/py/zetascale.svg" /></a>
</p>

# Benefits
- Write less code
- Prototype faster
- Bleeding-Edge Performance
- Reuseable Building Blocks
- Reduce Errors
- Scalability
- Build Models faster
- Full Stack Error Handling


# 🤝 Schedule a 1-on-1 Session
Book a [1-on-1 Session with Kye](https://calendly.com/apacai/agora), the Creator, to discuss any issues, provide feedback, or explore how we can improve Zeta for you.


## Installation

`pip install zetascale`

## Initiating Your Journey

Creating a model empowered with the aforementioned breakthrough research features is a breeze. Here's how to quickly materialize the renowned Flash Attention

```python
import torch
from zeta.nn.attention import FlashAttention

q = torch.randn(2, 4, 6, 8)
k = torch.randn(2, 4, 10, 8)
v = torch.randn(2, 4, 10, 8)

attention = FlashAttention(causal=False, dropout=0.1, flash=True)
output = attention(q, k, v)

print(output.shape) 

```

### ```RelativePositionBias```
- ```RelativePositionBias``` quantizes the distance between two positions into a certain number of buckets and then uses an embedding to get the relative position bias. This mechanism aids in the attention mechanism by providing biases based on relative positions between the query and key, rather than relying solely on their absolute positions.
```python
from zeta.nn import RelativePositionBias
import torch

# Initialize the RelativePositionBias module
rel_pos_bias = RelativePositionBias()

# Example 1: Compute bias for a single batch
bias_matrix = rel_pos_bias(1, 10, 10)

# Example 2: Utilize in conjunction with an attention mechanism
# NOTE: This is a mock example, and may not represent an actual attention mechanism's complete implementation.
class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.rel_pos_bias = RelativePositionBias()

    def forward(self, queries, keys):
        bias = self.rel_pos_bias(queries.size(0), queries.size(1), keys.size(1))
        # Further computations with bias in the attention mechanism...
        return None  # Placeholder

# Example 3: Modify default configurations
custom_rel_pos_bias = RelativePositionBias(bidirectional=False, num_buckets=64, max_distance=256, n_heads=8)

```

### `FeedForward`


# Documentation
[Click here for the documentation, it's at zeta.apac.ai](https://zeta.apac.ai)


## Contributing
- We need you to help us build the most re-useable, reliable, and high performance ML framework ever.

- [Check out the project board here!](https://github.com/users/kyegomez/projects/7/views/2)

- We need help writing tests and documentation!


# License 
- MIT