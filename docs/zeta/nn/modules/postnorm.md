# Module/Function Name: LayerNorm

The `PostNorm` class is a post-normalization module of `torch.nn.modules`. It applies layer normalization after the input is passed through a given module. The main objectives of this class are to improve the training stability of deep neural networks and to standardize the input to make the training less dependent on the scale of features.

Key features of `PostNorm` module:
- Post-normalization: Applies layer normalization after being passed through a given module.
- Dropout: Allows for the use of dropout probability on attention output weights.

### Class Definition
The `PostNorm` class has the following definition and parameters:

| Parameter  | Description  |
|---|---|
| dim  | The dimension of the input tensor  |
| fn  | The module to be applied to the input tensor  |

### Functionality and Usage
The `PostNorm` class performs a post-normalization on an input tensor using the given module. It applies layer normalization to the input tensor post application of `fn` module. The forward function `forward(x, **kwargs)` of the `PostNorm` module takes the input tensor `x` and additional keyword arguments `kwargs` to be passed to the underlying module.

#### Example 1: Usage within Model Architecture

```python
from torch import nn

from zeta.nn import PostNorm


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim):
        super().__init__()

        self.hidden_layer = nn.Linear(dim, hidden_dim)
        self.postnorm_layer = PostNorm(hidden_dim, nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = self.hidden_layer(x)
        output = self.postnorm_layer(x)

        return output


# Usage:
dim, hidden_dim, output_dim = 10, 20, 2
model = SimpleModel(dim, hidden_dim, output_dim)
inputs = torch.randn(64, dim)
outputs = model(inputs)

print(f"Input Shape: {inputs.shape}\nOutput Shape: {outputs.shape}")
```

#### Example 2: Usage with Image Data

```python
import torch
from torch import nn

from zeta.nn import PostNorm


# Define a model architecture for image data
class ImageModel(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.postnorm = PostNorm(output_dim, nn.ReLU())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.postnorm(x)


# Usage:
dim, hidden_dim, output_dim = 784, 256, 10  # Applicable for MNIST data
model = ImageModel(dim, hidden_dim, output_dim)
inputs = torch.randn(64, dim)
outputs = model(inputs)

print(f"Input Shape: {inputs.shape}\nOutput Shape: {outputs.shape}")
```

### Additional Information and Tips
- It is recommended to experiment with different input dimensions and types to understand the effect of post-normalization on model training.
- In case of errors or unexpected behavior, double-check the dimensions of the input tensor for compatibility with the post-normalization process.

### References and Resources
For further exploration into layer normalization in neural networks, the official documentation of PyTorch can be found at: [PyTorch Documentation on Layer Normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
