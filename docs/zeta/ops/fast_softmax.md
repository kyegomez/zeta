# fast_softmax

The `fast_softmax` function is a utility designed to compute the softmax of a given tensor in a numerically stable manner using the LogSumExp trick. The softmax function is a crucial component in many machine learning applications, especially those related to natural language processing and neural networks. It turns logits (i.e., raw output from a linear layer) into probabilities that sum up to 1.

Numerical instability can arise when dealing with large numbers due to overflow or underflow during the exponential operation in the traditional softmax calculation. The LogSumExp trick helps mitigate this issue by shifting the input values by their maximum value before the exponential operation.

This documentation provides thorough explanations, examples, and best practices to utilize the `fast_softmax` function effectively.

## Function Definition

`fast_softmax(tensor)`

### Parameters:

| Parameter | Type     | Description                                |
|-----------|----------|--------------------------------------------|
| `tensor`  | Tensor   | The input tensor for which to compute the softmax. |

### Returns:

A Tensor representing the softmax of the input tensor.

### Usage

The `fast_softmax` function can be used like a regular softmax function. However, it is particularly useful when the input tensor has high magnitude numbers and there is a risk of numerical overflow or underflow with a standard softmax implementation.

### Examples

#### Example 1: Basic usage

```python
import torch

from zeta.ops import fast_softmax

# Suppose we have an input tensor of logits
logits = torch.tensor([2.0, 1.0, 0.1])

# We apply fast_softmax to obtain the probabilities
probabilities = fast_softmax(logits)

print(probabilities)
```

#### Example 2: Large number handling

```python
import torch

from zeta.ops import fast_softmax

# When dealing with large numbers
large_logits = torch.tensor([12345.0, 67890.0, 1.0e5])

# Traditional softmax could fail due to numerical instability,
# but fast_softmax can handle this
probabilities = fast_softmax(large_logits)

print(probabilities)
```

#### Example 3: Batch processing

```python
import torch

from zeta.ops import fast_softmax

# Batch of logits
batch_logits = torch.rand(32, 10)  # Batch of 32 samples, each with 10 logits

# Compute softmax for the entire batch
batch_probabilities = fast_softmax(batch_logits)

print(batch_probabilities)
```

## Detailed Explanation

The `fast_softmax` function operates by first finding the maximum value in the input tensor and subtracting it from all elements in the tensor. This "shift" of the input tensor helps in reducing the likelihood of exponential values becoming too large. After applying the exponential function, the resultant tensor is then normalized by the sum of these exponentials, ensuring that all output values sum to 1, consistent with probability distributions.

### Numerical Stability: The LogSumExp Trick

The key to the numerical stability provided by the `fast_softmax` function lies in the LogSumExp trick. By shifting the inputs to have a maximum of zero before the exponential function is applied, we reduce the chances of reaching the floating-point overflow threshold. Since this shift does not change the relative differences between input values, it preserves the ratios necessary for accurate softmax computation.

## Common Issues and Solutions

- **Underflow and Overflow**: The most common issue addressed by `fast_softmax` is the numerical underflow and overflow during exponential calculations. By using `fast_softmax`, you should be able to avoid these issues even when dealing with input tensors containing large values.
  
- **Batch Processing**: When dealing with batches of data, ensure that the input tensor has the appropriate shape, where one dimension typically represents the batch size and the other represents the logits for each sample.

## References and Further Reading

For further exploration of the concepts behind the softmax function and the LogSumExp trick, the following resources may be helpful:

- [Bishop, Christopher M. "Pattern recognition and machine learning." (2006): 4-73](https://www.springer.com/gp/book/9780387310732)
- Goodfellow, Ian, et al. "Deep learning." MIT press, 2016.

