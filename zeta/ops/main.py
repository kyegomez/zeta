import enum
import logging
from typing import Tuple, Union, List
from einops import rearrange
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class NewtonConvergenceFlag(enum.Enum):
    REACHED_MAX_ITERS = 0
    CONVERGED = 1


class RootInvMethod(enum.Enum):
    EIGEN = 0
    NEWTON = 1


def check_diagonal(A: Tensor) -> Tensor:
    """Checks if symmetric matrix is diagonal."""

    A_shape = A.shape
    if len(A_shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")

    m, n = A_shape
    if m != n:
        raise ValueError("Matrix is not square!")

    return ~torch.any(A.reshape(-1)[:-1].reshape(m - 1, n + 1)[:, 1:].bool())


def matrix_inverse_root(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    exponent_multiplier: float = 1.0,
    root_inv_method: RootInvMethod = RootInvMethod.EIGEN,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    is_diagonal: Union[Tensor, bool] = False,
    retry_double_precision: bool = True,
) -> Tensor:
    """Computes matrix root inverse of square symmetric positive definite matrix.

    Args:
        A (Tensor): Square matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        exponent_multiplier (float): exponent multiplier in the eigen method (Default: 1.0)
        root_inv_method (RootInvMethod): Specifies method to use to compute root inverse. (Default: RootInvMethod.EIGEN)
        max_iterations (int): Maximum number of iterations for coupled Newton iteration. (Default: 1000)
        tolerance (float): Tolerance for computing root inverse using coupled Newton iteration. (Default: 1e-6)
        is_diagonal (Tensor, bool): Flag for whether or not matrix is diagonal. If so, will compute root inverse by computing
            root inverse of diagonal entries. (Default: False)
        retry_double_precision (bool): Flag for re-trying eigendecomposition with higher precision if lower precision fails due
            to CuSOLVER failure. (Default: True)

    Returns:
        X (Tensor): Inverse root of matrix A.

    """

    # check if matrix is scalar
    if torch.numel(A) == 1:
        alpha = torch.as_tensor(-exponent_multiplier / root)
        return (A + epsilon) ** alpha

    # check matrix shape
    if len(A.shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square!")

    if is_diagonal:
        X = matrix_root_diagonal(
            A=A,
            root=root,
            epsilon=epsilon,
            inverse=True,
            exponent_multiplier=exponent_multiplier,
            return_full_matrix=True,
        )
    elif root_inv_method == RootInvMethod.EIGEN:
        X, _, _ = _matrix_root_eigen(
            A=A,
            root=root,
            epsilon=epsilon,
            inverse=True,
            exponent_multiplier=exponent_multiplier,
            retry_double_precision=retry_double_precision,
        )
    elif root_inv_method == RootInvMethod.NEWTON:
        if exponent_multiplier != 1.0:
            raise ValueError(
                f"Exponent multiplier {exponent_multiplier} must be equal to 1"
                " to use coupled inverse Newton iteration!"
            )

        X, _, termination_flag, _, _ = _matrix_inverse_root_newton(
            A=A,
            root=root,
            epsilon=epsilon,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if termination_flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
            logging.warning(
                "Newton did not converge and reached maximum number of"
                " iterations!"
            )
    else:
        raise NotImplementedError(
            "Root inverse method is not implemented! Specified root inverse"
            " method is "
            + str(root_inv_method)
            + "."
        )

    return X


def matrix_root_diagonal(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    inverse: bool = True,
    exponent_multiplier: float = 1.0,
    return_full_matrix: bool = False,
) -> Tensor:
    """Computes matrix inverse root for a diagonal matrix by taking inverse square root of diagonal entries.

    Args:
        A (Tensor): One- or two-dimensional tensor containing either the diagonal entries of the matrix or a diagonal matrix.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        inverse (bool): Returns inverse root matrix. (Default: True)
        return_full_matrix (bool): Returns full matrix by taking torch.diag of diagonal entries. (bool: False)

    Returns:
        X (Tensor): Inverse root of diagonal entries.

    """

    # check order of tensor
    order = len(A.shape)
    if order == 2:
        A = torch.diag(A)
    elif order > 2:
        raise ValueError("Matrix is not 2-dimensional!")

    # check if root is positive integer
    if root <= 0:
        raise ValueError(f"Root {root} should be positive!")

    # compute matrix power
    alpha = exponent_multiplier / root
    if inverse:
        alpha = -alpha

    X = (A + epsilon).pow(alpha)
    return torch.diag(X) if return_full_matrix else X


def _matrix_root_eigen(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    inverse: bool = True,
    exponent_multiplier: float = 1.0,
    make_positive_semidefinite: bool = True,
    retry_double_precision: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute matrix (inverse) root using eigendecomposition of symmetric positive (semi-)definite matrix.

            A = Q L Q^T => A^{1/r} = Q L^{1/r} Q^T OR A^{-1/r} = Q L^{-1/r} Q^T

    Assumes matrix A is symmetric.

    Args:
        A (Tensor): Square matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        inverse (bool): Returns inverse root matrix. (Default: True)
        exponent_multiplier (float): exponent multiplier in the eigen method (Default: 1.0)
        make_positive_semidefinite (bool): Perturbs matrix eigenvalues to ensure it is numerically positive semi-definite. (Default: True)
        retry_double_precision (bool): Flag for re-trying eigendecomposition with higher precision if lower precision fails due
            to CuSOLVER failure. (Default: True)

    Returns:
        X (Tensor): (Inverse) root of matrix. Same dimensions as A.
        L (Tensor): Eigenvalues of A.
        Q (Tensor): Orthogonal matrix consisting of eigenvectors of A.

    """

    # check if root is positive integer
    if root <= 0:
        raise ValueError(f"Root {root} should be positive!")

    # compute matrix power
    alpha = exponent_multiplier / root
    if inverse:
        alpha = -alpha

    # compute eigendecomposition and compute minimum eigenvalue
    try:
        L, Q = torch.linalg.eigh(A)

    except Exception as exception:
        if retry_double_precision and A.dtype != torch.float64:
            logger.warning(
                f"Failed to compute eigendecomposition in {A.dtype} precision"
                f" with exception {exception}! Retrying in double precision..."
            )
            L, Q = torch.linalg.eigh(A.double())
        else:
            raise exception

    lambda_min = torch.min(L)

    # make eigenvalues >= 0 (if necessary)
    if make_positive_semidefinite:
        L += -torch.minimum(lambda_min, torch.as_tensor(0.0))

    # add epsilon
    L += epsilon

    # compute inverse preconditioner
    X = Q * L.pow(alpha).unsqueeze(0) @ Q.T

    return X, L, Q


def _matrix_inverse_root_newton(
    A,
    root: int,
    epsilon: float = 0.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[Tensor, Tensor, NewtonConvergenceFlag, int, Tensor]:
    """Compute matrix inverse root using coupled inverse Newton iteration.

        alpha <- -1 / p
        X <- 1/c * I
        M <- 1/c^p * A
        repeat until convergence
            M' <- (1 - alpha) * I + alpha * M
            X <- X * M'
            M <- M'^p * M

    where c = (2 |A|_F / (p + 1))^{1/p}. This ensures that |A|_2 <= |A|_F < (p + 1) c^p, which guarantees convergence.
    We will instead use z = (p + 1) / (2 * |A|_F).

    NOTE: Exponent multiplier not compatible with coupled inverse Newton iteration!

    Args:
        A (Tensor): Matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        max_iterations (int): Maximum number of iterations. (Default: 1000)
        tolerance (float): Tolerance. (Default: 1e-6)

    Returns:
        A_root (Tensor): Inverse square root of matrix.
        M (Tensor): Coupled matrix.
        termination_flag (NewtonConvergenceFlag): Specifies convergence.
        iteration (int): Number of iterations.
        error (Tensor): Final error between M and I.

    """

    # initialize iteration, dimension, and alpha
    iteration = 0
    dim = A.shape[0]
    alpha = -1 / root
    identity = torch.eye(dim, dtype=A.dtype, device=A.device)

    # add regularization
    A.add_(identity, alpha=epsilon)

    # initialize matrices
    A_nrm = torch.linalg.norm(A)
    z = (root + 1) / (2 * A_nrm)
    X = z ** (-alpha) * identity
    M = z * A
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `float`.
    error = torch.dist(M, identity, p=torch.inf)

    # main for loop
    while error > tolerance and iteration < max_iterations:
        iteration += 1
        M_p = M.mul(alpha).add_(identity, alpha=(1 - alpha))
        X = X @ M_p
        M = torch.linalg.matrix_power(M_p, root) @ M
        error = torch.dist(M, identity, p=torch.inf)

    # determine convergence flag
    termination_flag = (
        NewtonConvergenceFlag.CONVERGED
        if error <= tolerance
        else NewtonConvergenceFlag.REACHED_MAX_ITERS
    )

    return X, M, termination_flag, iteration, error


def compute_matrix_root_inverse_residuals(
    A: Tensor,
    X_hat: Tensor,
    root: int,
    epsilon: float,
    exponent_multiplier: float,
) -> Tuple[Tensor, Tensor]:
    """Compute residual of matrix root inverse for debugging purposes.

        relative error    = ||X - X_hat||_inf / ||X||_inf
        relative residual = ||A X^r - I||_inf

    Args:
        A (Tensor): Matrix of interest.
        X (Tensor): Computed matrix root inverse.
        root (int): Root of interest.
        epsilon (float): Adds epsilon * I to matrix.
        exponent_multiplier (float): Exponent multiplier to be multiplied to the numerator of the inverse root.

    Returns:
        absolute_error (Tensor): absolute error of matrix root inverse
        relative_error (Tensor): relative error of matrix root inverse
        residual (Tensor): residual of matrix root inverse

    """

    # check shape of matrix
    if len(A.shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square!")
    elif A.shape != X_hat.shape:
        raise ValueError("Matrix shapes do not match!")

    # compute error by comparing against double precision
    X = matrix_inverse_root(
        A.double(),
        root,
        epsilon=epsilon,
        exponent_multiplier=exponent_multiplier,
    )
    relative_error = torch.dist(X, X_hat, p=torch.inf) / torch.norm(
        X, p=torch.inf
    )

    # compute residual
    if exponent_multiplier == 1.0:
        X_invr = torch.linalg.matrix_power(X_hat.double(), n=-root)
    else:
        X_invr, _, _ = _matrix_root_eigen(
            X_hat.double(),
            root=1,
            epsilon=0.0,
            inverse=True,
            make_positive_semidefinite=True,
            exponent_multiplier=root / exponent_multiplier,
        )

    A_reg = A.double() + epsilon * torch.eye(
        A.shape[0], dtype=torch.float64, device=A.device
    )
    relative_residual = torch.dist(X_invr, A_reg, p=torch.inf) / torch.norm(
        A_reg, p=torch.inf
    )

    return relative_error, relative_residual


def merge_small_dims(tensor_shape: List[int], threshold: int) -> List[int]:
    """
    Reshapes tensor by merging small dimenions

    Args:
    tensor_shape (List[int]):  the shape of the tensor
    threshold(int) threshold on the maximum size of each dimension

    Returns:
        new_tensor_shape (List[int]) New tensor shape
    """

    new_tensor_shape = [tensor_shape[0]]
    for next_tensor_shape in tensor_shape[1:]:
        new_dimension = new_tensor_shape[-1] * next_tensor_shape
        if (
            new_tensor_shape[-1] == 1
            or next_tensor_shape == 1
            or new_dimension <= threshold
        ):
            new_tensor_shape[-1] = new_dimension
        else:
            new_tensor_shape.append(next_tensor_shape)

    return new_tensor_shape


def multi_dim_split(
    tensor: Tensor,
    splits: List[int],
) -> List[Tensor]:
    """
    Chunks tensor across multiple dimenions based on splits

    Args;
        tensor(tensor): gradient or tensor to split
        splits (List[int]) List of sizes for each block or chunk along each dimension

    Returns:
        split_grad(List[Tensor]): List of tensors
    """
    split_tensors = [tensor]
    for dim, split in enumerate(splits):
        split_tensors = [
            s for t in split_tensors for s in torch.split(t, split, dim=dim)
        ]
    return split_tensors


def multi_dim_cat(split_tensors: List[Tensor], num_splits: List[int]) -> Tensor:
    """Concatenates multiple tensors to form single tensor across multiple dimensions.

    Args:
        split_tensor (List[Tensor]): List of tensor splits or blocks.
        num_splits (List[int]): Number of splits/blocks.

    Returns:
        merged_tensor (Tensor): Merged tensor.

    """
    merged_tensor = split_tensors
    for dim, split in reversed(list(enumerate(num_splits))):
        if split > 0:
            merged_tensor = [
                torch.cat(merged_tensor[i : i + split], dim=dim)
                for i in range(0, len(merged_tensor), split)
            ]
    assert len(merged_tensor) == 1
    return merged_tensor[0]


def img_transpose(x):
    return rearrange(x, "b c h w -> b h w c")


def img_transpose_2daxis(x):
    return rearrange(x, "h w c -> w h c")


def img_composition_axis(x):
    return rearrange(x, "b h w c -> (b h) w c")


def img_compose_bw(x):
    return rearrange(x, "b h w c -> h (b w) c")


# decomposition of axis => inverse process, which represents an xis as a combination of new axis
# b1=2 is to decompose to 6 to b1=2 and b2=3
def img_decompose(x):
    return rearrange(x, "(b1 b2) h w c -> b1 b2 h w c", b1=2).shape


def img_compose_decompose(x):
    return rearrange(x, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=2)


# b1 is merged with width and b2 with height
def img_comp_decomp_merge(x):
    return rearrange(x, "(b1 b2) h w c -> (b2 h) (b1 w) c", b1=2)


# move part of width dimension to height
# width to height as image width shrunk by 2 and height doubled
def img_width_to_height(x):
    return rearrange(x, "b h (w w2) c -> (h w2) (b w) c", w2=2)


# order of axes
def img_order_of_axes(x):
    return rearrange(x, "b h w c -> h (b w) c")


# for each batch and for each pair of channels we sum over h and w
def gram_matrix_new(y):
    b, ch, h, w = y.shape
    return torch.einsum(
        "bchw,bdhw->bcd",
        [y, y],
    ) / (h * w)


# channel shuffle from shufflenet
def channel_shuffle_new(x, groups):
    return rearrange(
        x,
        "b (c1 c2) h w -> b (c2 c1) h w",
        c1=groups,
    )


# GLOW depth to space
def unsqueeze_2d_new(input, factor=2):
    return rearrange(
        input, "b (c h2 w2) h w -> b c (h h2) (w w2)", h2=factor, w2=factor
    )


def squeeze_2d_new(input, factor=2):
    return rearrange(
        input, "b c (h h2) (w w2) -> b (c h2 w2) h w", h2=factor, w2=factor
    )
