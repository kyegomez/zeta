import torch
import pytest
from zeta.nn.modules.s4 import s4d_kernel

# Test cases for s4d_kernel function


# Test 1: Basic test with valid inputs
def test_s4d_kernel_basic():
    A = torch.tensor([[1.0, 2.0, 3.0]])
    B = torch.tensor([[0.5, 1.0, 1.5]])
    C = torch.tensor([[0.2, 0.4, 0.6]])
    dt = 0.1
    L = 5
    result = s4d_kernel(A, B, C, dt, L)
    assert result.shape == (1, 5, 3)
    assert torch.allclose(
        result,
        torch.tensor(
            [
                [
                    [0.2, 0.4, 0.6],
                    [0.2602, 0.5488, 0.8617],
                    [0.3293, 0.6978, 1.0947],
                    [0.4072, 0.8661, 1.3574],
                    [0.4938, 1.0461, 1.6424],
                ]
            ]
        ),
        atol=1e-4,
    )


# Test 2: Test with incompatible tensor dimensions
def test_s4d_kernel_incompatible_dimensions():
    A = torch.tensor([[1.0, 2.0, 3.0]])
    B = torch.tensor([[0.5, 1.0, 1.5]])
    C = torch.tensor([[0.2, 0.4, 0.6]])
    dt = 0.1
    L = 5
    # Make A and B incompatible by adding an extra dimension to A
    A = A.unsqueeze(0)
    with pytest.raises(ValueError):
        s4d_kernel(A, B, C, dt, L)


# Test 3: Test with invalid data type for dt
def test_s4d_kernel_invalid_dt_type():
    A = torch.tensor([[1.0, 2.0, 3.0]])
    B = torch.tensor([[0.5, 1.0, 1.5]])
    C = torch.tensor([[0.2, 0.4, 0.6]])
    dt = "0.1"  # Should be a float, but provided as a string
    L = 5
    with pytest.raises(TypeError):
        s4d_kernel(A, B, C, dt, L)


# Test 4: Test with invalid data type for L
def test_s4d_kernel_invalid_L_type():
    A = torch.tensor([[1.0, 2.0, 3.0]])
    B = torch.tensor([[0.5, 1.0, 1.5]])
    C = torch.tensor([[0.2, 0.4, 0.6]])
    dt = 0.1
    L = 5.5  # Should be an integer, but provided as a float
    with pytest.raises(TypeError):
        s4d_kernel(A, B, C, dt, L)


# Test 5: Test with zero-dimensional tensors
def test_s4d_kernel_zero_dimensional_tensors():
    A = torch.tensor(1.0)
    B = torch.tensor(0.5)
    C = torch.tensor(0.2)
    dt = 0.1
    L = 5
    result = s4d_kernel(A, B, C, dt, L)
    assert result.shape == (1, 5, 1)
    assert torch.allclose(
        result,
        torch.tensor([[[0.2], [0.2], [0.2], [0.2], [0.2]]]),
        atol=1e-4,
    )


# Add more test cases as needed...
