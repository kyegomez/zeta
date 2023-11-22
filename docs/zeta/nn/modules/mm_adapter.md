# Module: MultiModalAdapterDenseNetwork

The `MultiModalAdapterDenseNetwork` module is designed for creating multi-modal adapter dense networks in PyTorch. It allows you to build deep neural networks with skip connections for efficient multi-modal data processing.

### Overview

In multi-modal data processing, combining information from different sources or modalities is crucial. This module provides a flexible way to design such networks by stacking multiple layers, applying normalization, activation functions, and skip connections.

### Class Definition

```python
class MultiModalAdapterDenseNetwork(nn.Module):
    """
    Multi-modal adapter dense network that takes a tensor of shape (batch_size, dim) and returns a tensor of shape (batch_size, dim).

    Flow:
    x -> norm -> linear 1 -> silu -> concatenate -> linear 2 -> skip connection -> output

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension.
        depth (int): The depth of the network.
        activation (nn.Module): The activation function.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: The forward pass of the network.
    """
```

### Parameters

| Parameter       | Description                                             | Data Type | Default Value |
|-----------------|---------------------------------------------------------|-----------|---------------|
| dim             | The input dimension.                                    | int       | None          |
| hidden_dim      | The hidden dimension.                                   | int       | None          |
| depth           | The depth of the network.                               | int       | None          |
| activation      | The activation function.                                | nn.Module | nn.SiLU()     |

### Forward Method

```python
def forward(x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the network.
    """
```

### How It Works

The `MultiModalAdapterDenseNetwork` class works by stacking multiple layers of neural network operations, including normalization, linear transformations, activation functions, concatenation, and skip connections. Here's how it operates step by step:

1. Input tensor `x` is first normalized using layer normalization.
2. Two linear transformations are applied to `x`: `linear 1` and `linear 2`.
3. The activation function `silu` is applied to the output of `linear 1`.
4. The output of `linear 1` and `linear 2` is concatenated.
5. The result is passed through the `skip_connections` module, which combines it with the original input tensor `x`.
6. The final output is obtained.

### Usage Examples

#### Example 1: Creating and Using the Network

```python
import torch
from torch import nn
from zeta.nn import MultiModalAdapterDenseNetwork

# Create an instance of MultiModalAdapterDenseNetwork
mm_adapter = MultiModalAdapterDenseNetwork(
    dim=512,
    hidden_dim=1024,
    depth=3,
)

# Generate a random input tensor
x = torch.randn(1, 512)

# Perform a forward pass
output = mm_adapter(x)

# Print the output shape
print(output.shape)  # Output shape: torch.Size([1, 1024, 512])
```

In this example, we create an instance of `MultiModalAdapterDenseNetwork`, pass an input tensor through it, and print the output shape.

#### Example 2: Custom Activation Function

```python
import torch
from torch import nn
from zeta.nn import MultiModalAdapterDenseNetwork

# Define a custom activation function
class CustomActivation(nn.Module):
    def forward(self, x):
        return x * 2

# Create an instance of MultiModalAdapterDenseNetwork with the custom activation
mm_adapter = MultiModalAdapterDenseNetwork(
    dim=512,
    hidden_dim=1024,
    depth=3,
    activation=CustomActivation(),
)

# Generate a random input tensor
x = torch.randn(1, 512)

# Perform a forward pass
output = mm_adapter(x)
```

In this example, we create a custom activation function and use it when creating an instance of `MultiModalAdapterDenseNetwork`.

#### Example 3: Custom Depth and Hidden Dimension

```python
import torch
from torch import nn
from zeta.nn import MultiModalAdapterDenseNetwork

# Create an instance of MultiModalAdapterDenseNetwork with custom depth and hidden dimension
mm_adapter = MultiModalAdapterDenseNetwork(
    dim=512,
    hidden_dim=2048,  # Increased hidden dimension
    depth=5,           # Increased depth
)

# Generate a random input tensor
x = torch.randn(1, 512)

# Perform a forward pass
output = mm_adapter(x)
```

In this example, we create an instance of `MultiModalAdapterDenseNetwork` with custom depth and hidden dimension values.

### Additional Information and Tips

- The `MultiModalAdapterDenseNetwork` class allows you to experiment with different architectures and activation functions for multi-modal data processing.
- You can customize the activation function by providing your own module as the `activation` argument.
- Experiment with different values for `dim`, `hidden_dim`, and `depth` to find the optimal architecture for your task.

This documentation provides a comprehensive guide to the `MultiModalAdapterDenseNetwork` module, including its purpose, parameters, usage examples, and tips for customization. Feel free to explore and adapt this module to suit your specific multi-modal data processing needs.

### References and Resources

- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- Multi-modal Data Processing Techniques: [https://arxiv.org/abs/2107.15912](https://arxiv.org/abs/2107.15912) (Reference paper for multi-modal data processing)
- [Paper Origination: M2UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models](https://arxiv.org/pdf/2311.11255.pdf)