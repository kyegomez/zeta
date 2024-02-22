`FilmConditioning` Module

Introduction:
The FilmConditioning module applies feature-wise affine transformations to the input tensor, conditioning it based on a conditioning tensor. This module is particularly useful in scenarios where feature-based conditioning is required in convolutional neural network architectures.

Args:
Number of channels (int): Specifies the number of channels in the input tensor.

Attributes:
num_channels (int): Number of channels in the input tensor.
projection_add (nn.Linear): Linear layer for additive projection.
projection_mult (nn.Linear): Linear layer for multiplicative projection.

Class Definition:
```python
class FilmConditioning(nn.Module):
    def __init__(self, num_channels: int, *args, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self._projection_add = nn.Linear(num_channels, num_channels)
        self._projection_mult = nn.Linear(num_channels, num_channels)
```

Functionality and Usage:
The `__init__` method initializes the module and its attributes. Two linear layers are defined for additive and multiplicative projections of conditioning. The `forward` method applies affine transformations to the input tensor based on the conditioning tensor.
```python
def forward(self, conv_filters: torch.Tensor, conditioning: torch.Tensor):
    projected_cond_add = self._projection_add(conditioning)
    projected_cond_mult = self._projection_mult(conditioning)
    # Modifying the result is based on the conditioning tensor
    return result
```

Usage Examples:

Usage Example 1: Applying Film Conditioning
```python
import torch
import torch.nn as nn

from zeta.nn import FilmConditioning

# Define input tensors
conv_filters = torch.randn(10, 3, 32, 32)
conditioning = torch.randn(10, 3)

# Create an instance of FilmConditioning
film_conditioning = FilmConditioning(3)

# Applying film conditioning
result = film_conditioning(conv_filters, conditioning)
print(result.shape)
```

Usage Example 2: Applying Film Conditioning for another example
```python
import torch
import torch.nn as nn

from zeta.nn import FilmConditioning

# Define input tensors
conv_filters = torch.randn(5, 4, 20, 20)
conditioning = torch.randn(5, 4)

# Create an instance of FilmConditioning
film_conditioning = FilmConditioning(4)

# Applying film conditioning
result = film_conditioning(conv_filters, conditioning)
print(result.shape)
```

Usage Example 3: Usage Example
```python
import torch
import torch.nn as nn

from zeta.nn import FilmConditioning

# Define input tensors
conv_filters = torch.randn(8, 2, 50, 50)
conditioning = torch.randn(8, 2)

# Create an instance of FilmConditioning
film_conditioning = FilmConditioning(2)

# Applying film conditioning
result = film_conditioning(conv_filters, conditioning)
print(result.shape)
```

References and Resources:
Expected format for the documentation should be provided here for any references.
