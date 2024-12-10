import torch  # Use standard Python import for PyTorch
cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_sin(input_tensor):
    cdef int i
    cdef int size = input_tensor.numel()

    # Convert the PyTorch tensor to a NumPy array
    np_array = input_tensor.numpy()

    # Apply sin element-wise using NumPy
    np_output = np.sin(np_array)

    # Convert the NumPy array back to a PyTorch tensor
    output_tensor = torch.from_numpy(np_output)

    return output_tensor
