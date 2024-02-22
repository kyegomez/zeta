# _matrix_root_eigen


The principal function within the zeta.ops library is `_matrix_root_eigen`, which computes the (inverse) root of a given symmetric positive (semi-)definite matrix using eigendecomposition. The computation is based on the relation `A = Q * L * Q^T`, where `A` is the initial matrix, `Q` is a matrix of eigenvectors, and `L` is a diagonal matrix with eigenvalues. This function is particularly useful in applications such as signal processing, quantum mechanics, and machine learning, where matrix root computations are often required.


The `_matrix_root_eigen` function is the cornerstone of the zeta.ops library. Its purpose is to calculate the root or inverse root of a matrix by decomposing it into its eigenvectors and eigenvalues, modifying the eigenvalues as per the desired operation (root or inverse root), and then reconstructing the matrix.

## Architecture of `_matrix_root_eigen`

The `_matrix_root_eigen` function is built upon PyTorch's linear algebra capabilities and follows a clear sequence of steps:

1. Verify if the root is a positive integer.
2. Calculate the power to which the eigenvalues need to be raised (`alpha`).
3. Perform eigendecomposition on the input matrix `A`.
4. Modify the eigenvalues to ensure they are positive if the `make_positive_semidefinite` flag is set.
5. Add a small `epsilon` value if necessary to ensure numerical stability.
6. Compute the (inverse) root matrix using the modified eigenvalues and the eigenvectors.

This architecture ensures that even matrices that might have numerical stability issues or slightly negative eigenvalues due to floating-point errors can be handled gracefully.

## `_matrix_root_eigen`: Method Signature

Below is the method signature for the `_matrix_root_eigen` function, alongside an explanation of its arguments and returned values:

| Argument                   | Type      | Default Value         | Description                                                                         |
|----------------------------|-----------|-----------------------|-------------------------------------------------------------------------------------|
| A                          | Tensor    | Required              | The square matrix of interest.                                                      |
| root                       | int       | Required              | The root of interest, which should be a natural number.                             |
| epsilon                    | float     | 0.0                   | A small value added to the matrix to avoid numerical instability.                   |
| inverse                    | bool      | True                  | If set to True, the function returns the inverse root matrix; otherwise, the root.  |
| exponent_multiplier        | float     | 1.0                   | A multiplier applied to the eigenvalue exponent in the root calculation.            |
| make_positive_semidefinite | bool      | True                  | Perturbs eigenvalues to ensure the matrix is positive semi-definite.                |
| retry_double_precision     | bool      | True                  | Retries eigendecomposition with higher precision if initial attempt fails.         |

Returns:

| Returned Value | Type    | Description                                                                         |
|----------------|---------|-------------------------------------------------------------------------------------|
| X              | Tensor  | The computed (inverse) root of matrix A.                                            |
| L              | Tensor  | Eigenvalues of matrix A.                                                            |
| Q              | Tensor  | Orthogonal matrix consisting of eigenvectors of matrix A.                           |

## Usage Examples

In the following sections, we'll look at three different ways to use the `_matrix_root_eigen` function from the zeta.ops library, along with the required imports and full example code.

### Example 1: Basic Matrix Root Calculation

In this example, we'll calculate the square root of a 2x2 symmetric positive definite matrix.

```python
import torch

from zeta.ops import _matrix_root_eigen

# Define a 2x2 symmetric positive definite matrix
A = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

# Calculate the square root of the matrix
X, L, Q = _matrix_root_eigen(A, root=2)

print("Matrix A:\n", A)
print("Square Root of A:\n", X)
```

### Example 2: Matrix Inverse Root with Epsilon Perturbation

In this example, an `epsilon` perturbation is added for numerical stability, and the inverse square root is calculated.

```python
import torch

from zeta.ops import _matrix_root_eigen

# Define a 3x3 symmetric positive definite matrix
A = torch.tensor([[4.0, 2.0, 0.0], [2.0, 4.0, 1.0], [0.0, 1.0, 3.0]])

# Calculate the inverse square root of the matrix, adding epsilon for stability
X, L, Q = _matrix_root_eigen(A, root=2, epsilon=1e-5, inverse=True)

print("Matrix A:\n", A)
print("Inverse Square Root of A with Epsilon:\n", X)
```

### Example 3: High-Precision Calculation with Positive Semi-Definite Guarantee

This example demonstrates a more robust usage where the calculation is attempted in high precision, and the function ensures the matrix is positive semi-definite before computing its root.

```python
import torch

from zeta.ops import _matrix_root_eigen

# Define a 3x3 symmetric positive semi-definite matrix with potential numerical issues
A = torch.tensor([[1e-5, 0.0, 0.0], [0.0, 5.0, 4.0], [0.0, 4.0, 5.0]])

# Calculate the square root, ensuring positive semi-definiteness and retrying in double precision if needed
X, L, Q = _matrix_root_eigen(
    A, root=2, make_positive_semidefinite=True, retry_double_precision=True
)

print("Matrix A:\n", A)
print("Square Root with Positive Semi-Definite Guarantee:\n", X)
```

## Additional Remarks

When using the `_matrix_root_eigen` function, keep in mind that it assumes the input matrix `A` is symmetric. If the matrix is not symmetric, the results will not be valid. Also, use caution when setting the `epsilon` value to ensure that it does not distort the accurate computation of the matrix root more than necessary for numerical stability.

## Conclusion

The zeta.ops library, specifically the `_matrix_root_eigen` function, is a powerful tool for scientific computation, providing advanced functionality for matrix root operations using eigendecomposition. By understanding the parameters and utilizing the provided examples, users can effectively leverage this functionality for their research or computational needs.

## References and Further Reading

To learn more about the mathematical operations used in this library, consult the following resources:

- "Numerical Linear Algebra" by Lloyd N. Trefethen and David Bau, III.
- "Matrix Analysis" by Rajendra Bhatia.
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html

