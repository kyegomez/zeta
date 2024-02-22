# DynamicPositionBias Documentation

---

## **Overview and Introduction**

The `DynamicPositionBias` class from the `zeta` library is designed to compute positional biases dynamically based on relative distances between positions in a sequence. This module can be crucial in attention mechanisms where relative position matters, as commonly seen in Transformers.

Key concepts:
- **Relative Position**: The difference in position between two tokens in a sequence.
- **Positional Bias**: A bias introduced based on the relative position, to indicate how two positions are related.
- **MLP (Multi-Layer Perceptron)**: A type of feedforward neural network consisting of multiple layers of nodes in a directed graph.

## **Class Definition**

```python
class DynamicPositionBias(nn.Module):
    def __init__(self, dim: int, heads: int): ...
```

### Parameters:
- `dim` (`int`): The dimension of the intermediary layer in the MLP.
- `heads` (`int`): The number of attention heads. This also dictates the output dimension of the bias.

### Attributes:
- `mlp` (`nn.Sequential`): Multi-Layer Perceptron used to compute the bias based on relative distance.
  
## **Functionality and Usage**

### Method: `forward(i: int, j: int) -> torch.Tensor`
Computes the positional bias based on the relative distance between positions `i` and `j`.

#### Parameters:
- `i` (`int`): Starting position in the sequence.
- `j` (`int`): Ending position in the sequence.

#### Returns:
- `bias` (`torch.Tensor`): A tensor representing the bias, of shape `(heads, i, j)`.

#### Usage:

The positional bias can be utilized in attention mechanisms to provide awareness of relative position between tokens.

### Examples:

1. **Basic Usage**:
    ```python
    import torch

    from zeta import DynamicPositionBias

    # Initialize the module
    module = DynamicPositionBias(dim=64, heads=8)

    # Compute bias for positions 0 to 5
    bias = module(0, 5)
    ```

2. **Integration with Transformer**:
    ```python
    import torch
    from torch.nn import MultiheadAttention

    from zeta import DynamicPositionBias


    class CustomAttention(MultiheadAttention):
        def __init__(self, embed_dim, num_heads):
            super().__init__(embed_dim, num_heads)
            self.pos_bias = DynamicPositionBias(dim=embed_dim, heads=num_heads)

        # Override the forward method to include positional bias
        # ... (implementation details)
    ```

3. **Inspecting the Bias**:
    ```python
    import matplotlib.pyplot as plt
    import torch

    from zeta import DynamicPositionBias

    # Initialize the module
    module = DynamicPositionBias(dim=64, heads=8)

    # Compute bias and visualize for positions 0 to 5
    bias = module(0, 5)
    plt.imshow(bias[0].detach().numpy())
    plt.show()
    ```

## **Additional Information and Tips**

- Ensure that `j >= i` when calling the forward method.
- The model relies on the `einops` library for tensor rearrangement. Ensure you have this dependency installed.
- This module primarily assists in capturing the relative positional information between two positions in a sequence. It might be beneficial when absolute positional embeddings are not available or not preferred.

## **References and Resources**
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - Introduces the concept of attention mechanisms that can benefit from positional information.
- [Einops Documentation](https://github.com/arogozhnikov/einops) - For tensor rearrangement operations used in the implementation.

---

## Mathematical Representation:

Given a sequence from `i` to `j`:

\[ S = [s_i, s_{i+1}, ... s_{j-1}] \]

The relative distance \( R \) for any two elements \( s_x \) and \( s_y \) from this sequence is:

\[ R(x, y) = |x - y| \]

The bias for a specific head `h` and relative distance \( r \) can be represented as:

\[ \text{bias}_h(r) = MLP_h(r) \]

Where `MLP_h` is the Multi-Layer Perceptron specific to head `h`.