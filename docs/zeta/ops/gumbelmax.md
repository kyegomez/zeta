# gumbelmax


`GumbelMax` serves the purpose of providing a differentiable approximation to the process of drawing samples from a categorical distribution. This is particularly useful in areas such as reinforcement learning or generative models where the Gumbel-Max trick can be used to sample actions or categories without losing gradient information.

#### Parameters:

| Parameter | Type    | Default | Description                                                      |
|-----------|---------|---------|------------------------------------------------------------------|
| `x`       | Tensor  | N/A     | The input tensor containing unnormalized log probabilities.      |
| `temp`    | float   | 1.0     | The temperature parameter controlling the sharpness of the distribution.     |
| `hard`    | boolean | False   | Determines the output format: one-hot encoded vector or probabilities distribution. |

#### Description:
The `GumbelMax` function manipulates the input tensor `x` by adding Gumbel noise to generate samples from a Gumbel distribution. This process serves as an approximation to sampling from a categorical distribution. When the `hard` parameter is set to `True`, the output is a one-hot encoded tensor representing the selected category. Otherwise, a probability distribution tensor is returned. The `temp` parameter affects the 'sharpness' of the softmax output; lower values make the output closer to one-hot encoding.

### Functionality and Usage

`GumbelMax` utilizes the Gumbel-Max trick, which enables gradient-based optimization over discrete variables by providing a continuous representation that can be used in backpropagation. The function first creates Gumbel noise and adds it to the input tensor, then applies a softmax function to generate a probability distribution over possible classes. The temperature parameter `temp` controls the concentration of the distribution – a smaller `temp` leads to a more concentrated, 'sharper' distribution, which makes the output resemble a one-hot tensor more closely.

The `hard` parameter allows users to decide between a 'soft', probabilistic representation and a 'hard', deterministic one (one-hot encoded). Even with the hard version, gradients can still flow through the operation during backpropagation due to the straight-through estimator trick employed.

### Usage Examples

#### Example 1: Soft Sampling

```python
import torch
import torch.nn.functional as F
from zeta.ops import gumbelmax

# Unnormalized log probabilities
logits = torch.tensor([[0.1, 0.5, 0.4]])

# Soft sampling with default temperature
soft_sample = gumbelmax(logits)
print(soft_sample)
```

#### Example 2: Hard Sampling

```python
# Hard sampling with temperature t=0.5
hard_sample = gumbelmax(logits, temp=0.5, hard=True)
print(hard_sample)
```

#### Example 3: Changing Temperature

```python
# Soft sampling with a higher temperature, resulting in a smoother distribution
smooth_sample = gumbelmax(logits, temp=5.0)
print(smooth_sample)

# Soft sampling with a lower temperature, resulting in a sharper distribution
sharp_sample = gumbelmax(logits, temp=0.1)
print(sharp_sample)
```

### Additional Information and Tips

- The Gumbel-Max trick is a cornerstone technique for non-differentiable sampling processes, making them compatible with gradient-based optimization techniques.
- Keep an eye on the temperature parameter as it can significantly affect the behavior of the function, especially the variance of the samples drawn.
- While using `hard=True` provides a deterministic output, the gradients can still be computed due to the reparameterization trick employed internally.

