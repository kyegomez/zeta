# top_a

# Module: zeta.utils

## Function: top_a()

## Description
This utility function, `top_a()`, is an implementation of a technique known as 'Top-K filtering' or 'Nucleus sampling'. 
It involves softmaxing the logits and selecting a subset of it whose cumulative probability exceeds a certain threshold. It is particularly useful in natural language processing tasks to refine the output of language models. 

The function takes a tensor of logits, applies a softmax function for normalization, associates these probabilities with a certain limit, and then applies a filter to modify the logits based on the associated limit.

## Parameters 

| Parameter  | Type                  | Description                                                    |
|------------|-----------------------|----------------------------------------------------------------|
| logits     | PyTorch Tensor        | The input tensor for which the softmax will be computed.       |
| min_p_pow  | float (Optional)      | The minimal power to which max probability is raised. Default is 2.0.        |
| min_p_ratio| float (Optional)      | The minimal ratio to minimum power used to set the limit. Default is 0.02. |

## Returns
This function returns a modified version of the input tensor, logits with respect to the specified limit.

## Code

```python
import torch
import torch.nn.functional as F


def top_a(logits, min_p_pow=2.0, min_p_ratio=0.02):
    # compute softmax probabilities
    probs = F.softmax(logits, dim=-1)

    # set limit with respect to maximum probabily and min_p_pow and min_p_ratio
    limit = torch.pow(torch.max(probs), min_p_pow) * min_p_ratio

    # apply filter to modify the logits with respect to the limit
    logits[probs < limit] = float("-inf")
    logits[probs >= limit] = 1
    return logits
```

## Examples

### EXAMPLE 1

In this example, we'll compute the top_a function on a tensor of logits.

```python
import torch

from zeta.utils import top_a

# Create a tensor of logits
logits = torch.tensor([0.1, 0.2, 0.3, 0.4])

# Call the function
result = top_a(logits)

# Output
print(result)
```

### EXAMPLE 2

In this example, we use user-defined minimum power `min_p_pow` and minimum ratio `min_p_ratio`.

```python
import torch

from zeta.utils import top_a

# Create a tensor of logits
logits = torch.tensor([0.1, 0.5, 0.2, 0.4])

# Call the function
result = top_a(logits, min_p_pow=3.0, min_p_ratio=0.01)

# Output
print(result)
```

### EXAMPLE 3

In this example, we see how changing the `min_p_pow` affects the output.

```python
import torch

from zeta.utils import top_a

# Create a tensor of logits
logits = torch.tensor([0.2, 0.3, 0.5, 0.5])

# Call the function with different min_p_pow values
result1 = top_a(logits, min_p_pow=1.0)
result2 = top_a(logits, min_p_pow=2.0)
result3 = top_a(logits, min_p_pow=3.0)

# Output
print(result1)
print(result2)
print(result3)
``` 

## Note

Deep learning practitioners should maintain a good practice of casting tensors into the right device (CPU or GPU) before operations. Ensure the logits tensor is on the right device before calling `top_a()`. Additionally, the values in the tensor should be in logits (unnormalized scores or predictions) and not in the form of probabilities (i.e., no softmax has been applied). 

This function is meant to be a utility. For a more specialized task, slight modifications may be required as per the use case. Thus, it should not be considered as a one-size-fits-all solution, but rather as a template code for selecting samples contingent upon a specific set of probabilities.
