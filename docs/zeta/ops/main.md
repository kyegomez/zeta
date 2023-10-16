# `zeta.ops` Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Module Overview](#module-overview)
4. [Function Definitions](#function-definitions)
   - [`check_diagonal`](#check_diagonal)
   - [`matrix_inverse_root`](#matrix_inverse_root)
   - [`matrix_root_diagonal`](#matrix_root_diagonal)
   - [`_matrix_root_eigen`](#_matrix_root_eigen)
   - [`_matrix_inverse_root_newton`](#_matrix_inverse_root_newton)
   - [`compute_matrix_root_inverse_residuals`](#compute_matrix_root_inverse_residuals)
   - [`merge_small_dims`](#merge_small_dims)
   - [`multi_dim_split`](#multi_dim_split)
   - [`multi_dim_cat`](#multi_dim_cat)
5. [Usage Examples](#usage-examples)
   - [Example 1: Matrix Inverse Root using Eigen Decomposition](#example-1-matrix-inverse-root-using-eigen-decomposition)
   - [Example 2: Matrix Inverse Root using Coupled Newton Iteration](#example-2-matrix-inverse-root-using-coupled-newton-iteration)
   - [Example 3: Residual Computation](#example-3-residual-computation)
   - [Example 4: Merging Small Dimensions](#example-4-merging-small-dimensions)
   - [Example 5: Multi-Dimensional Split and Concatenation](#example-5-multi-dimensional-split-and-concatenation)

---

### 1. Introduction <a name="introduction"></a>

Welcome to the `zeta.ops` documentation! `zeta.ops` is a Python library designed to handle matrix operations related to matrix inverse root computation and other related tasks. It provides functionality for computing matrix inverse roots, working with diagonal matrices, performing eigen decomposition, and more.

This documentation will guide you through the installation process, provide an overview of the module, and explain the various functions and their usage with examples.

### 2. Installation <a name="installation"></a>

You can install zeta using the Python package manager pip:

```bash
pip install zetascale
```

Once installed, you can import the library in your Python scripts and start using its functionality.

### 3. Module Overview <a name="module-overview"></a>

`zeta.ops` primarily focuses on matrix operations and computations related to matrix inverse roots. Here are some key features of the zeta library:

- Computing matrix inverse roots of square symmetric positive definite matrices.
- Handling diagonal matrices efficiently.
- Eigen decomposition of symmetric matrices.
- Coupled Newton iteration for matrix inverse root computation.
- Residual computation for debugging purposes.
- Merging small dimensions in tensor shapes.
- Multi-dimensional splitting and concatenation of tensors.

In the following sections, we'll delve into the details of each function and provide examples of their usage.

### 4. Function Definitions <a name="function-definitions"></a>

#### 4.1 `check_diagonal`

```python
check_diagonal(A: Tensor) -> Tensor
```

Checks if a symmetric matrix is diagonal.

- `A` (Tensor): The input matrix to check.

Returns:
- `Tensor`: A boolean tensor indicating whether the matrix is diagonal.

#### 4.2 `matrix_inverse_root`

```python
matrix_inverse_root(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    exponent_multiplier: float = 1.0,
    root_inv_method: RootInvMethod = RootInvMethod.EIGEN,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    is_diagonal: Union[Tensor, bool] = False,
    retry_double_precision: bool = True,
) -> Tensor
```

Computes the matrix inverse root of a square symmetric positive definite matrix.

- `A` (Tensor): The input matrix.
- `root` (int): The root of interest (any natural number).
- `epsilon` (float): Adds epsilon * I to the matrix before taking the matrix root (default: 0.0).
- `exponent_multiplier` (float): Exponent multiplier in the eigen method (default: 1.0).
- `root_inv_method` (RootInvMethod): Specifies the method to use for computing the root inverse (default: RootInvMethod.EIGEN).
- `max_iterations` (int): Maximum number of iterations for coupled Newton iteration (default: 1000).
- `tolerance` (float): Tolerance for computing the root inverse using coupled Newton iteration (default: 1e-6).
- `is_diagonal` (Union[Tensor, bool]): Flag for whether or not the matrix is diagonal. If set to True, the function computes the root inverse of diagonal entries (default: False).
- `retry_double_precision` (bool): Flag for re-trying eigendecomposition with higher precision if lower precision fails due to CuSOLVER failure

 (default: True).

Returns:
- `Tensor`: The matrix inverse root.

#### 4.3 `matrix_root_diagonal`

```python
matrix_root_diagonal(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    inverse: bool = True,
    exponent_multiplier: float = 1.0,
    return_full_matrix: bool = False,
) -> Tensor
```

Computes the matrix inverse root for a diagonal matrix by taking the inverse square root of diagonal entries.

- `A` (Tensor): A one- or two-dimensional tensor containing either the diagonal entries of the matrix or a diagonal matrix.
- `root` (int): The root of interest (any natural number).
- `epsilon` (float): Adds epsilon * I to the matrix before taking the matrix root (default: 0.0).
- `inverse` (bool): If True, returns the inverse root matrix (default: True).
- `exponent_multiplier` (float): Exponent multiplier to be multiplied to the numerator of the inverse root (default: 1.0).
- `return_full_matrix` (bool): If True, returns the full matrix by taking the torch.diag of diagonal entries (default: False).

Returns:
- `Tensor`: The inverse root of diagonal entries.

#### 4.4 `_matrix_root_eigen`

```python
_matrix_root_eigen(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    inverse: bool = True,
    exponent_multiplier: float = 1.0,
    make_positive_semidefinite: bool = True,
    retry_double_precision: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]
```

Compute the matrix (inverse) root using eigendecomposition of a symmetric positive (semi-)definite matrix.

- `A` (Tensor): The square matrix of interest.
- `root` (int): The root of interest (any natural number).
- `epsilon` (float): Adds epsilon * I to the matrix before taking the matrix root (default: 0.0).
- `inverse` (bool): If True, returns the inverse root matrix (default: True).
- `exponent_multiplier` (float): Exponent multiplier in the eigen method (default: 1.0).
- `make_positive_semidefinite` (bool): Perturbs matrix eigenvalues to ensure it is numerically positive semi-definite (default: True).
- `retry_double_precision` (bool): Flag for re-trying eigendecomposition with higher precision if lower precision fails due to CuSOLVER failure (default: True).

Returns:
- `Tuple[Tensor, Tensor, Tensor]`: A tuple containing the (inverse) root matrix, eigenvalues, and orthogonal matrix consisting of eigenvectors.

#### 4.5 `_matrix_inverse_root_newton`

```python
_matrix_inverse_root_newton(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[Tensor, Tensor, NewtonConvergenceFlag, int, Tensor]
```

Compute the matrix inverse root using coupled inverse Newton iteration.

- `A` (Tensor): The matrix of interest.
- `root` (int): The root of interest (any natural number).
- `epsilon` (float): Adds epsilon * I to the matrix before taking the matrix root (default: 0.0).
- `max_iterations` (int): Maximum number of iterations (default: 1000).
- `tolerance` (float): Tolerance (default: 1e-6).

Returns:
- `Tuple[Tensor, Tensor, NewtonConvergenceFlag, int, Tensor]`: A tuple containing the (inverse) root matrix, coupled matrix, convergence flag, number of iterations, and final error.

#### 4.6 `compute_matrix_root_inverse_residuals`

```python
compute_matrix_root_inverse_residuals(
    A: Tensor,
    X_hat: Tensor,
    root: int,
    epsilon: float,
    exponent_multiplier: float,
) -> Tuple[Tensor, Tensor]
```

Compute the residual of the matrix root inverse for debugging purposes.

- `A` (Tensor): The matrix of interest.
- `X_hat` (Tensor): The computed matrix root inverse.
- `root` (int): The root of interest.
- `epsilon` (float): Adds epsilon * I to the matrix.
- `exponent_multiplier` (float): Exponent multiplier to be multiplied to the numerator of the inverse root.

Returns:
- `Tuple[Tensor, Tensor]`: A tuple containing the absolute error and relative error of the matrix root inverse.

#### 4.7 `merge_small_dims`

```python
merge_small_dims(
    tensor_shape: List[int],
    threshold: int
) -> List[int]
```

Reshapes a tensor by merging small dimensions.

- `tensor_shape` (List[int]): The shape of the tensor.
- `threshold` (int): The threshold on the maximum size of each dimension.

Returns:
- `List[int]`: The new tensor shape.

#### 4.8 `multi_dim_split`

```python
multi_dim_split(
    tensor: Tensor,
    splits: List[int],
) -> List[Tensor]
```

Chunks a tensor across multiple dimensions based on splits.

- `tensor` (Tensor): The gradient or tensor to split.
- `splits` (List[int]): The list of sizes for each block or chunk along each dimension.

Returns:
- `List[Tensor]`: The list of tensors after splitting.

#### 4.9 `multi_dim_cat`

```python
multi_dim_cat(
    split_tensors: List[Tensor],
    num_splits: List[int]
) -> Tensor
```

Concatenates multiple tensors to form a single tensor across multiple dimensions.

- `split_tensors` (List[Tensor]): The list of tensor splits or blocks.
- `num_splits` (List[int]): The number of splits/blocks.

Returns:
- `Tensor`: The merged tensor.

### 5. Usage Examples <a name="usage-examples"></a>

Let's explore some usage examples of the functions provided by the zeta library.

#### 5.1 Example 1: Matrix Inverse Root using

 Eigen Method

In this example, we will compute the matrix inverse root of a symmetric positive definite matrix using the eigen method. We will use the following parameters:

```python
import torch
from zeta import matrix_inverse_root, RootInvMethod

A = torch.tensor([[4.0, 2.0], [2.0, 3.0]])
root = 2
epsilon = 1e-6
exponent_multiplier = 1.0
method = RootInvMethod.EIGEN

X = matrix_inverse_root(A, root, epsilon=epsilon, exponent_multiplier=exponent_multiplier, root_inv_method=method)
print(X)
```
#### 5.2 Example 2: Matrix Root Diagonal

In this example, we will compute the matrix inverse root for a diagonal matrix by taking the inverse square root of diagonal entries. We will use the following parameters:

```python
import torch
from zeta import matrix_root_diagonal

A = torch.tensor([4.0, 9.0])
root = 2
epsilon = 1e-6
exponent_multiplier = 1.0

X = matrix_root_diagonal(A, root, epsilon=epsilon, exponent_multiplier=exponent_multiplier)
print(X)
```

#### 5.3 Example 3: Matrix Inverse Root using Newton Method

In this example, we will compute the matrix inverse root using the coupled inverse Newton iteration method. We will use the following parameters:

```python
import torch
from zeta import matrix_inverse_root, RootInvMethod

A = torch.tensor([[4.0, 2.0], [2.0, 3.0]])
root = 2
epsilon = 1e-6
exponent_multiplier = 1.0
method = RootInvMethod.NEWTON

X = matrix_inverse_root(A, root, epsilon=epsilon, exponent_multiplier=exponent_multiplier, root_inv_method=method)
print(X)
```

In this example, we compute the matrix inverse root of a 2x2 matrix `A` with a root of 2 using the coupled inverse Newton iteration method. The result is a matrix `X` that represents the inverse square root of `A`.

### 6. Additional Information and Tips <a name="additional-information"></a>

Here are some additional tips and information to help you effectively use the zeta library:

- **Exponent Multiplier**: The `exponent_multiplier` parameter is used to adjust the exponent in the eigen method. Be cautious when changing this parameter, as it may affect the correctness of the result.

- **Eigen Method Precision**: The eigen method may fail in lower precision (e.g., float32) for certain matrices. You can enable `retry_double_precision` to retry the eigendecomposition in double precision if the initial attempt fails.

- **Newton Method Convergence**: The coupled inverse Newton iteration method may not always converge within the specified `max_iterations` and `tolerance`. It's important to monitor the convergence and adjust these parameters accordingly.

- **Debugging with Residuals**: The function `compute_matrix_root_inverse_residuals` can be used to compute residuals for debugging purposes. It helps in verifying the correctness of the computed matrix root inverse.

### 7. References and Resources <a name="references"></a>

Here are some references and resources for further exploration of matrix root and inverse methods:

- [Matrix Square Root](https://en.wikipedia.org/wiki/Square_root_of_a_matrix)
- [Matrix Inverse Root](https://en.wikipedia.org/wiki/Matrix_power#Matrix_nth_roots_and_matrix_logarithm)
- [Eigenvalue Decomposition](https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html)

For more advanced use cases and research, consider exploring academic papers and textbooks on linear algebra and matrix computations.