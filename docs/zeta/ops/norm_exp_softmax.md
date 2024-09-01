# norm_exp_softmax


This documentation provides a comprehensive guide on how to use the `norm_exp_softmax` function, which is part of the `zeta.ops` library module. The function is designed to apply a normalized exponential softmax to input tensors, scaling the exponentiation as specified. The goal is to transform the input tensor into a probability distribution where each element represents a probability that corresponds to its input value after scaling.

## Overview of `norm_exp_softmax`

### Purpose

The `norm_exp_softmax` function implements a stable version of the softmax operation, which is largely used in machine learning, especially in the context of classification tasks and attention mechanisms. It is designed to map a vector of real numbers into a probability distribution. The function provides an option to scale the input before exponentiation, which might assist in adjusting the sharpness of the probability distribution.

### Functionality

The function computes the softmax of the input tensor by exponentiating each element, scaling it by a given factor, and then normalizing the results so that they sum to 1. This creates a new tensor where the values represent probabilities.

### Architecture

Under the hood, `norm_exp_softmax` employs the `torch.exp` function to compute the exponential of each element in the tensor and normalizes the values along the specified dimension, usually the last dimension.

The architecture is designed to ensure numerical stability by directly computing the exponential of the scaled tensor and dividing by its sum in one go, rather than separately computing the exponential, sum and then division. This helps prevent overflow or underflow in the exponential function by scaling down large numbers before exponentiation.

## `norm_exp_softmax` Function Definition

```python
def norm_exp_softmax(x, scale=1.0):
    # See inline description
```

### Parameters

| Parameter | Type      | Description                                        | Default |
|-----------|-----------|----------------------------------------------------|---------|
| `x`       | Tensor    | The input tensor whose softmax is to be computed.  | N/A     |
| `scale`   | float     | The scale parameter to adjust the sharpness of the softmax distribution. | 1.0     |

### Expected Behavior

When `norm_exp_softmax` is called, it expects a tensor as input and an optional scaling factor. It will apply the softmax function to the input tensor, scaling each element in the tensor before exponentiation, and ensure that the final result is a tensor of the same size where the elements sum up to 1 along the last dimension.

## How to Use `norm_exp_softmax`

### Basic Usage Example

```python
import torch

from zeta.ops import norm_exp_softmax

# Input tensor
x = torch.tensor([1.0, 2.0, 3.0])

# Apply norm_exp_softmax without scaling
softmax_probs = norm_exp_softmax(x)

print(softmax_probs)  # Output will be a probability distribution tensor
```

### Usage Example with Scaling

```python
import torch

from zeta.ops import norm_exp_softmax

# Input tensor
x = torch.tensor([1.0, 2.0, 3.0])

# Apply norm_exp_softmax with scaling
scale_factor = 0.5
softmax_probs_scaled = norm_exp_softmax(x, scale=scale_factor)

print(
    softmax_probs_scaled
)  # Output will be a softly scaled probability distribution tensor
```

### Advanced Usage Example

```python
import torch

from zeta.ops import norm_exp_softmax

# Input tensor with batch dimension
x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]])

# Apply norm_exp_softmax with scaling across batched input
scale_factor = 2.0
batch_softmax_probs = norm_exp_softmax(x, scale=scale_factor)

print(batch_softmax_probs)  # Output will be a batch of probability distribution tensors
```

## Additional Information and Tips

- It is important to choose the `scale` parameter carefully as it may dramatically change the behavior of the softmax function. A larger `scale` makes the softmax function "peakier" (i.e., more confident), while a lower `scale` makes it smoother (i.e., more uniform).
- The softmax function is widely used as the final step in classification models to interpret the logits (raw model outputs) as probabilities.
- The `norm_exp_softmax` operation assumes that input tensors are unbatched by default. If tensors are batched, the operation is applied independently to each batch.

## Conclusion and Further Reading

The `norm_exp_softmax` function is an essential component in many machine learning pipelines, providing a way to interpret and manipulate raw model outputs as probabilities. By ensuring numerical stability and providing a scaling option, it offers both reliability and flexibility for a wide range of applications.

For deeper insights into the softmax function and its applications, consider referring to the following resources:
- [PyTorch Official Documentation](https://pytorch.org/docs/stable/nn.html#torch.nn.Softmax)
- The `torch.nn.functional.softmax` function documentation for understanding comparisons and different ways to use softmax in PyTorch.
- [Deep Learning Book by Ian Goodfellow and Yoshua Bengio and Aaron Courville](https://www.deeplearningbook.org/) for a more theoretical perspective on softmax in the context of deep learning.

Remember, practice is key to understanding the nuances of the softmax function and its applications. Experiment with different scales and problem domains to truly grasp its utility and impact.
