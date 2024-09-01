# `softmaxes` in `zeta.ops`

## Overview

The `zeta.ops` library is a collection of various softmax operations, each tailored to specific use cases and computational needs. From traditional softmax to sparse softmax and logit-scaled variants, this library offers a wide array of activation functions for deep learning practitioners.

Softmax functions are essential components of many deep learning models, especially those dealing with classification tasks. The functions in this library allow for the customization of the softmax behavior, enabling users to fine-tune models to specific requirements.

## Methods

Below are the methods provided by the `zeta.ops` library:

### 1. Standard Softmax
- **Function**: `standard_softmax(tensor)`
- **Description**: Computes the standard softmax function.
- **Parameters**: 
  - `tensor`: Input tensor.
- **Returns**: Softmax-applied tensor.

### 2. SELU Softmax
- **Function**: `selu_softmax(x)`
- **Description**: Applies the SELU activation function to the tensor before computing the softmax.
- **Parameters**: 
  - `x`: Input tensor.
- **Returns**: SELU and softmax-applied tensor.

### 3. Sparsemax
- **Function**: `sparsemax(x, k)`
- **Description**: Computes the sparsemax function, retaining only the top `k` values.
- **Parameters**: 
  - `x`: Input tensor.
  - `k`: Number of elements to retain.
- **Returns**: Sparsemax-applied tensor.

### 4. Local Softmax
- **Function**: `local_softmax(tensor, num_chunks)`
- **Description**: Splits the tensor into chunks and applies softmax locally to each chunk.
- **Parameters**: 
  - `tensor`: Input tensor.
  - `num_chunks`: Number of chunks to split the tensor into.
- **Returns**: Concatenated tensor after local softmax application.

### 5. Fast Softmax
- **Function**: `fast_softmax(tensor)`
- **Description**: Computes softmax using the LogSumExp trick for numerical stability.
- **Parameters**: 
  - `tensor`: Input tensor.
- **Returns**: Softmax-applied tensor.

### 6. Sparse Softmax
- **Function**: `sparse_softmax(z, k)`
- **Description**: Computes softmax while retaining only the top `k` values.
- **Parameters**: 
  - `z`: Input tensor.
  - `k`: Number of elements to retain.
- **Returns**: Sparse softmax-applied tensor.

### 7. Gumbelmax
- **Function**: `gumbelmax(x, temp, hard)`
- **Description**: Applies Gumbel noise to the tensor and computes softmax.
- **Parameters**: 
  - `x`: Input tensor.
  - `temp`: Temperature parameter.
  - `hard`: Boolean; if True, returns a one-hot tensor, otherwise a probability distribution.
- **Returns**: Softmax-applied tensor with Gumbel noise.

### 8. Softmax with Temperature
- **Function**: `temp_softmax(x, temp)`
- **Description**: Scales the tensor using a temperature parameter before computing softmax.
- **Parameters**: 
  - `x`: Input tensor.
  - `temp`: Temperature parameter.
- **Returns**: Temperature-scaled softmax tensor.

### 9. Logit Scaled Softmax
- **Function**: `logit_scaled_softmax(x, scale)`
- **Description**: Multiplies the tensor by a scale factor before computing softmax.
- **Parameters**: 
  - `x`: Input tensor.
  - `scale`: Scale parameter.
- **Returns**: Logit-scaled softmax tensor.

### 10. Norm Exponential Softmax
- **Function**: `norm_exp_softmax(x, scale)`
- **Description**: Applies the normalized exponential function to the tensor.
- **Parameters**: 
  - `x`: Input tensor.
  - `scale`: Scale parameter.
- **Returns**: Normalized exponential softmax tensor.

## Usage Examples

Here are some usage examples for each method:

```python
import torch

from zeta.ops import selu_softmax, standard_softmax

# Sample tensor
tensor = torch.tensor([2.0, 1.0, 0.1])

# 1. Standard Softmax
output = standard_softmax(tensor)
print(output)

# 2. SELU Softmax
output = selu_softmax(tensor)
print(output)

# ... [Continue for all methods]
```

Replace the function name with the desired method and adjust the parameters accordingly for other examples.

---

**Note**: Always ensure the input tensor's dimensions match the expected input for each function. Some functions like sparsemax require additional parameters, so be sure to provide them.