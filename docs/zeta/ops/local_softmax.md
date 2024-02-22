# local_softmax


The `local_softmax` function from the `zeta.ops` library is designed to handle softmax computations on large inputs by dividing them into smaller, more manageable chunks. This can be particularly useful for tasks that involve processing very large tensors that may not fit into memory if softmax were applied to the entire tensor at once.

## Overview and Introduction

Softmax is a mathematical function commonly used in the fields of machine learning and deep learning, particularly in classification tasks. It turns a vector of raw scores, often called logits, into probabilities by exponentiating and normalizing the input values. However, when dealing with very large inputs, performing softmax on the entire dataset at once can be computationally expensive and memory-intensive.

The `local_softmax` function alleviates this concern by dividing the input tensor into multiple chunks, applying softmax individually on each chunk, and then concatenating the results together. This allows for more efficient memory usage and can reduce the computational overhead when dealing with large input tensors.

## Function Definition

| Parameter   | Description                                           | Type   | Default Value |
|-------------|-------------------------------------------------------|--------|---------------|
| tensor      | The input tensor on which softmax will be applied.    | Tensor | -             |
| num_chunks  | The number of chunks to split the input tensor into.  | int    | 2             |

### `local_softmax` Function
```python
def local_softmax(tensor, num_chunks: int = 2):
    """
    Performs softmax on chunks of the input tensor.

    Parameters:
    - tensor (Tensor): The input tensor to be softmaxed.
    - num_chunks (int): Number of chunks the input tensor is split into.

    Returns:
    - Tensor: Concatenated tensor with applied softmax on each chunk.
    """
    # Implementation
```

## Functionality and Usage

The `local_softmax` function operates by splitting the input tensor along the zeroth dimension (rows) into the specified number of chunks. It then applies the softmax function, as provided by `torch.nn.functional.softmax`, to each chunk individually. Afterward, the function concatenates the softmaxed chunks back together along the same dimension to produce the final output tensor.

### Expected Inputs and Outputs
- **Input**: A tensor of any shape that can be split into the specified number of chunks along the zeroth dimension.
- **Output**: A tensor of the same shape as the input, where softmax has been applied to each corresponding chunk of the input.

### Usage Examples

Below are three usage examples illustrating how to use the `local_softmax` function with different inputs and chunk sizes.

#### Example 1: Basic Usage
```python
import torch
from torch.nn import functional as F

# Importing the local_softmax function
from zeta.ops import local_softmax

# Example tensor (for demonstration purposes)
input_tensor = torch.tensor([[2.0, 1.0], [0.5, -1.0], [1.0, 3.0], [2.0, 5.0]])

# Apply local_softmax with 2 chunks
output_tensor = local_softmax(input_tensor, num_chunks=2)
print(output_tensor)
```

#### Example 2: Using a Larger Number of Chunks
```python
import torch
from torch.nn import functional as F

# Importing the local_softmax function
from zeta.ops import local_softmax

# Another example with a larger tensor
large_input_tensor = torch.randn(10, 5)

# Apply local_softmax with 5 chunks
output_tensor = local_softmax(large_input_tensor, num_chunks=5)
print(output_tensor)
```

#### Example 3: Exception Handling When Number of Chunks Mismatch
```python
import torch
from torch.nn import functional as F

# Importing the local_softmax function
from zeta.ops import local_softmax

# Another example with tensor that can't be evenly split into chunks
odd_sized_tensor = torch.randn(7, 3)

# Attempt to apply local_softmax with 4 chunks
try:
    output_tensor = local_softmax(odd_sized_tensor, num_chunks=4)
    print(output_tensor)
except RuntimeError as e:
    print(f"Error: {e}")
```

Note: In the third example, since the input tensor cannot be evenly split into 4 chunks, a `RuntimeError` is raised by PyTorch. Users will need to handle such exceptions or ensure that the number of chunks divides the size of the first dimension of the tensor.

## Additional Information and Tips

- Ensure that the number of chunks specified in `num_chunks` is a divisor of the size of the tensor's zeroth dimension to avoid runtime errors.
- Consider the implications of performing softmax on chunks—that is, softmax will be applied independently to each chunk, not across the whole tensor. This means that if there is any relationship between the chunks that needs to be preserved, this method might not be appropriate.
- The choice of chunk size could potentially impact the performance of subsequent operations on the softmaxed tensor, so it may require some experimentation to find the optimal balance between memory usage and computational efficiency.

## References and Resources

For more information on the softmax function and its applications, the following resources may be useful:
- [PyTorch Documentation: `torch.nn.functional.softmax`](https://pytorch.org/docs/stable/nn.functional.html#softmax)
- [Stanford University's CS231n Notes on Softmax](http://cs231n.github.io/linear-classify/#softmax)
- [Understanding the Softmax Function by Sebastian Ruder](https://sebastianruder.com/softmax/)

These resources provide a deeper understanding of the theoretical background behind softmax and its implementation details within the PyTorch framework.
