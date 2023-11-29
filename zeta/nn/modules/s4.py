import torch
from typing import Tuple

def s4d_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, dt: float, L: int) -> torch.Tensor:
    """
    Compute the S4D convolution kernel for state space models on 3D tensors with shape (batch_size, seqlen, dim).

    Parameters:
    A (torch.Tensor): A tensor of shape (batch_size, dim) containing the eigenvalues of the state update matrix.
    B (torch.Tensor): A tensor of shape (batch_size, dim) containing the input-to-state weights.
    C (torch.Tensor): A tensor of shape (batch_size, dim) containing the state-to-output weights.
    dt (float): A scalar that represents the time step in the discrete-time SSM.
    L (int): The length of the sequence over which the convolution will be performed.

    Returns:
    torch.Tensor: A tensor of shape (batch_size, seqlen, dim) that represents the convolution of the inputs through the SSM.

    Raises:
    ValueError: If the dimensions of A, B, or C are not compatible.
    TypeError: If dt is not a float or L is not an integer.
    """

    # Ensure A, B, and C have the same size in the last dimension and compatible batch dimensions
    if A.size(-1) != B.size(-1) or A.size(-1) != C.size(-1) or A.shape[:-1] != B.shape[:-1] or A.shape[:-1] != C.shape[:-1]:
        raise ValueError("The last dimension of tensors A, B, and C must match and have compatible batch dimensions.")
    
    # Check that dt is a float and L is an integer
    if not isinstance(dt, float):
        raise TypeError("The time step dt must be a float.")
    if not isinstance(L, int):
        raise TypeError("The sequence length L must be an integer.")

    # Create a range of values from 0 to L-1 and reshape for broadcasting
    arange_L = torch.arange(L, dtype=A.dtype, device=A.device).view(L, 1)

    # Expand A and B for broadcasting with the sequence length
    A_expanded = A.unsqueeze(1)  # Shape: (batch_size, 1, dim)
    B_expanded = B.unsqueeze(1)  # Shape: (batch_size, 1, dim)

    # Perform the convolution kernel operation with proper broadcasting
    vandermonde = torch.exp(arange_L * dt * A_expanded)  # Shape: (seqlen, batch_size, dim)
    result = torch.sum(vandermonde * B_expanded * (torch.exp(dt * A_expanded) - 1) / A_expanded, dim=0)
    result = C.unsqueeze(1) * result  # Shape: (batch_size, seqlen, dim)

    return result

# # Example usage with random tensors:
# torch.manual_seed(0)  # For reproducibility
# batch_size = 5  # Example batch size
# N = 10  # Size of the state space
# L = 100  # Sequence length

# # Randomly generated tensors for A, B, and C with the correct shape and a random float for dt
# A_random = torch.randn(batch_size, N)
# B_random = torch.randn(batch_size, N)
# C_random = torch.randn(batch_size, N)
# dt_random = float(torch.rand(1).item())

# # Call the s4d_kernel function with the random tensors and parameters
# output = s4d_kernel(A_random, B_random, C_random, dt_random, L)
# print("Output of the s4d_kernel with random inputs:", output)
