# matrix_root_diagonal


```python
def matrix_root_diagonal(
    A: torch.Tensor,
    root: int,
    epsilon: float = 0.0,
    inverse: bool = True,
    exponent_multiplier: float = 1.0,
    return_full_matrix: bool = False
) -> torch.Tensor:
```
Computes the inverse root of a diagonal matrix by taking the inverse square root of the diagonal entries. This function can either manipulate the given tensor directly if it represents a diagonal of a matrix or extract the diagonal from a 2D tensor and then proceed with the computation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | `torch.Tensor` | | A tensor representing either the diagonal of a matrix or a full diagonal matrix. |
| `root` | `int` | | The root of interest. Must be a natural number. |
| `epsilon` | `float` | `0.0` | A small value added to the diagonal to avoid numerical issues. |
| `inverse` | `bool` | `True` | Specifies whether to return the inverse root. |
| `exponent_multiplier` | `float` | `1.0` | Multiplier for the exponent, providing additional transformation control. |
| `return_full_matrix` | `bool` | `False` | If `True`, the result is a full matrix with the diagonal altered. Otherwise, only the diagonal is returned. |

#### Returns

| Name | Type | Description |
|------|------|-------------|
| `X` | `torch.Tensor` | The resulting tensor after computing the inverse root of the diagonal matrix. |

#### Overview

The `matrix_root_diagonal` function is an essential utility for operations such as whitening a covariance matrix where the matrix root is needed. It supports both direct diagonal input and square matrices, giving it versatility for various use cases.

#### Architecture and Operation

The internal workflow checks the dimensionality of the input tensor `A`. It raises an exception for non-2D tensors. For input representing a full square matrix, it extracts the diagonal. The necessary inverse root computations are then applied to the diagonal entries, with an option to reintegrate them into a full matrix.

#### Usage Example 1: Basic Diagonal Tensor

```python
import torch

from zeta.ops import matrix_root_diagonal

# Create a diagonal tensor
A = torch.tensor([4.0, 9.0, 16.0])

# Compute the inverse square root of the diagonal
root_matrix = matrix_root_diagonal(A, root=2)

print(root_matrix)
```

#### Usage Example 2: Full matrix with epsilon

```python
import torch

from zeta.ops import matrix_root_diagonal

# Create a diagonal matrix
A = torch.diag(torch.tensor([4.0, 9.0, 16.0]))

# Compute the inverse square root of the diagonal with epsilon
root_matrix = matrix_root_diagonal(A, root=2, epsilon=0.1)

print(root_matrix)
```

#### Usage Example 3: Return Full Matrix

```python
import torch

from zeta.ops import matrix_root_diagonal

# Create a diagonal tensor
A = torch.tensor([4.0, 9.0, 16.0])

# Compute the inverse square root and return the full matrix
root_matrix = matrix_root_diagonal(A, root=2, return_full_matrix=True)

print(root_matrix)
```

#### Additional Information & Tips

- The function ensures numerical stability by adding a small value `epsilon` to the diagonal before computation.
- The computation involves element-wise operations. Hence, the input tensor `A` is expected to have one or two dimensions only.
- Setting `inverse` to `False` results in the computation of the direct root rather than the inverse.

#### References and Further Reading

For a better understanding of matrix roots and their applications, the following resources may be helpful:
- Higham, Nicholas J. "Computing real square roots of a real matrix." Linear Algebra and its applications 88 (1987): 405-430.
- Wikipedia entry on Matrix Functions: https://en.wikipedia.org/wiki/Matrix_function
