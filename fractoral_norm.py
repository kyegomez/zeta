from zeta.nn import FractoralNorm  # Importing the FractoralNorm class from the zeta.nn module
import torch  # Importing the torch module for tensor operations

# Norm
x = torch.randn(2, 3, 4)  # Generating a random tensor of size (2, 3, 4)

# FractoralNorm
normed = FractoralNorm(4, 4)(x)  # Applying the FractoralNorm operation to the tensor x

print(normed)  # Printing the size of the resulting tensor, which should be torch.Size([2, 3, 4])