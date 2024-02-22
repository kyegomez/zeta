# sparse_softmax

# Zeta Operations Library Documentation

## Module: `zeta.ops`

The `zeta.ops` module offers a specialized implementation of the `sparse_softmax` operation, which represents a differentiable and sparse alternative to the traditional softmax function. Designed for PyTorch, this module caters to situations where a sparse subset of activations is desired. This may be particularly useful in attention mechanisms where only the top-k values need to be considered while the rest are set to zero, hence promoting sparsity.

The `sparse_softmax` function is vital in scenarios where interpretability and model sparsity are of high concern. By concentrating the probability mass on a fixed number of elements and leaving the others explicitly zero, sparsemax facilitates a clear and discernible selection of features or tokens, which is invaluable for tasks such as natural language processing and feature selection.

## Sparse Softmax Function Definition

The `sparse_softmax` function accepts an input tensor and a specified number of elements (k) and applies a projection operation that maps the input onto the simplex of the same dimension in such a way that at most k components are non-zero.

### Parameters:

| Parameter | Type   | Description                                        | Default |
|-----------|--------|----------------------------------------------------|---------|
| `z`       | Tensor | The input tensor.                                  | ------  |
| `k`       | int    | The number of elements to keep while ensuring sparsity.| 3       |

### Functionality and Usage

The `sparse_softmax` function processes its input using a simple algorithm:

1. It sorts the input tensor `z` in descending order.
2. It applies the transformation `sparsemax(z) = max(0, z - tau(z))` where `tau(z) = (sum_i=1^k z_i - 1) / k` to the sorted tensor.

Below we provide detailed examples illustrating how to use the `sparse_softmax` function in three different scenarios.

### Example 1: Basic Usage

```python
import torch

from zeta.ops import sparse_softmax

# Define an input tensor
input_tensor = torch.tensor([2.0, 1.5, 0.1, -1.0, 3.2, 0.7], dtype=torch.float32)

# Apply sparse softmax with k = 3
output_tensor = sparse_softmax(input_tensor, k=3)

print(output_tensor)
```

In this basic example, an input tensor is defined with six elements. The `sparse_softmax` function is applied with `k=3`, indicating that only the top 3 activations will be considered while others will be zero.

### Example 2: Working with Batched Inputs

```python
import torch

from zeta.ops import sparse_softmax

# Define a batched input tensor
batched_input = torch.tensor(
    [[2.0, -0.5], [1.5, -1.0], [0.1, 2.5], [-1.0, 3.0]], dtype=torch.float32
)

# Apply sparse softmax to each sample in the batch with k = 2
batched_output = torch.stack([sparse_softmax(sample, k=2) for sample in batched_input])

print(batched_output)
```

In the second example, a batch of input tensors is defined. Each sample in the batch is independently processed with `sparse_softmax` with `k=2`.

### Example 3: Integration with Neural Network Layers

```python
import torch
import torch.nn as nn

from zeta.ops import sparse_softmax


class SparseAttention(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, queries, keys, values):
        # Compute the dot product between queries and keys
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))

        # Apply the sparse softmax to the attention scores
        sparse_attention_probs = torch.stack(
            [sparse_softmax(sample, k=self.k) for sample in attention_scores]
        )

        # Use the attention probabilities to weight the values
        weighted_values = torch.bmm(sparse_attention_probs, values)

        return weighted_values


# Example input tensors for the attention mechanism
queries = torch.randn(2, 3, 5)  # (batch_size, seq_length, model_dim)
keys = torch.randn(2, 3, 5)
values = torch.randn(2, 3, 5)

# Define our SparseAttention layer with k=2
sparse_attn_layer = SparseAttention(k=2)

# Pass through the attention layer
output_tensor = sparse_attn_layer(queries, keys, values)

print(output_tensor)
```

The third example illustrates the application in a neural network context, particularly within an attention mechanism. `SparseAttention` is defined as a network layer that applies `sparse_softmax` to the attention scores.

### Additional Information and Tips

The `sparse_softmax` function is differentiable, which allows it to be used seamlessly within deep learning architectures. While designed for use with PyTorch, the core idea can be adapted for other machine learning frameworks that support automatic differentiation.

Using the `sparse_softmax` function can lead to computational efficiencies, especially when the tensor's dimensionality is large but `k` remains small. Additionally, this promotes a form of interpretability as the non-zero elements in the output directly correspond to the top-k features deemed most important by the model.

### Common Issues and Recommendations

1. **Selection of k**: Choosing a proper `k` value is crucial for balancing sparsity and performance. A small `k` increases sparsity but might neglect important features. Conversely, a large `k` may dilute the attention mechanism's effectiveness.
2. **Batch Processing**: When working with batches, ensure that the sparse softmax operation is applied individually to each example to maintain the context of each sample.
3. **Gradients**: Sparse operations can possess gradients that differ from their dense counterparts. Keep a watchful eye on gradient flow during backpropagation, especially when integrating `sparse_softmax` in custom layers or loss functions.

### References and Resources

- For the theory behind sparse operations in neural networks and their implications in machine learning, refer to the paper "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification" by André F. T. Martins and Ramón Fernandez Astudillo.
- Additional readings and resources on sparsity in deep learning:
  - "Exploring Sparsity in Recurrent Neural Networks" by Sharan Narang et al.
  - "Deep Learning with Sparse Transformers" by Rewon Child et al.

The `sparse_softmax` function in the `zeta.ops` module offers a powerful and concise solution for imparting explicit sparsity within neural networks. Its utility in selective attention and feature extraction scenarios makes it an invaluable addition to the arsenal of operations available for PyTorch practitioners.
