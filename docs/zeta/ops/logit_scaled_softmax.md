# logit_scaled_softmax


The `zeta.ops` library is a collection of custom operations that augment the capabilities of PyTorch, a deep learning framework widely used for building neural networks. The primary goal of `zeta.ops` is to provide specialized and optimized operations that are not directly available within the standard PyTorch package, thereby enhancing the performance and functionality of PyTorch models.

## logit_scaled_softmax

### Definition

The `logit_scaled_softmax` function is a modified version of the standard softmax operation. It scales the logits before applying the softmax function, which can be useful in scenarios where control over the distribution sharpness of the output probabilities is desired.

### Parameters

| Parameter | Type    | Description                                        | Default Value |
| --------- | ------- | -------------------------------------------------- | ------------- |
| `x`       | Tensor  | The input tensor containing logits to be scaled.   | N/A           |
| `scale`   | float   | The scale parameter to adjust the sharpness.       | 1.0           |

### Function Description

```python
import torch.nn.functional as F

def logit_scaled_softmax(x, scale=1.0):
    """
    Computes the scaled softmax of the input tensor.

    Args:
        x (Tensor): The input tensor containing logits.
        scale (float, optional): A scaling factor to apply to logits before the softmax. Default: 1.0
    
    Returns:
        Tensor: A tensor containing the resulting scaled softmax probabilities.
    """
    return F.softmax(x * scale, dim=-1)
```

### Usage Examples

#### Example 1: Basic Usage

```python
import torch
from zeta.ops import logit_scaled_softmax

# Create a tensor of logits
logits = torch.tensor([1.0, 2.0, 3.0])

# Apply logit_scaled_softmax without scaling (default behavior)
softmax_probs = logit_scaled_softmax(logits)
print(softmax_probs)
```

#### Example 2: Adjusting Sharpness with Scale

```python
import torch
from zeta.ops import logit_scaled_softmax

# Create a tensor of logits
logits = torch.tensor([1.0, 2.0, 3.0])

# Apply logit_scaled_softmax with scaling to increase sharpness
scale = 2.0
sharper_softmax_probs = logit_scaled_softmax(logits, scale)
print(sharper_softmax_probs)
```

#### Example 3: Using logit_scaled_softmax in Neural Networks

```python
import torch
import torch.nn as nn
from zeta.ops import logit_scaled_softmax

# Define a simple neural network with logit_scaled_softmax
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 3)
    
    def forward(self, x, scale=1.0):
        logits = self.fc(x)
        return logit_scaled_softmax(logits, scale)

# Create a random input tensor
input_tensor = torch.randn(5, 10)

# Instantiate the neural network
model = SimpleNN()

# Forward pass with custom softmax operation
output_probs = model(input_tensor, scale=1.5)
print(output_probs)
```

### Functionality and Architecture

The `logit_scaled_softmax` function is designed to modulate the sharpness of the output probabilities obtained from the softmax function. Scaling logits prior to applying the softmax can be particularly useful when adjusting the confidence of the predictions made by a model.

Multiplying the logits by a scale factor greater than 1 increases the difference between the highest and other logits, leading to a sharper probability distribution where one class's probability is much higher than the others. Conversely, a scale factor less than 1 will make the probability distribution softer, providing a more uniform distribution of probabilities across classes.

This operation can be used in various parts of a neural network, such as the final classification layer or within attention mechanisms to control the distribution of attention weights.

### Additional Tips

- When using `logit_scaled_softmax`, experiment with different scale values as part of hyperparameter tuning to find the optimal level of sharpness for your specific use case.
- Be cautious when applying very high scale factors, as this might lead to numerical instability due to the softmax function's exponential nature.
- The `logit_scaled_softmax` is differentiable, allowing it to be incorporated into a model's architecture and trained end-to-end using backpropagation.

### References and Resources

- PyTorch Documentation: [Softmax Function](https://pytorch.org/docs/stable/nn.functional.html#softmax)
- Goodfellow, Ian, et al. "Deep Learning." MIT Press, 2016, section on softmax function, provides an in-depth background on the softmax function and its properties.

To explore more about PyTorch and deep learning models, consider visiting the official [PyTorch website](https://pytorch.org) and reviewing the extensive documentation and tutorials available.
