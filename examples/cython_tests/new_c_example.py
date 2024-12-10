import torch
import torch_extension  # Import the compiled Cython module

# Create a PyTorch tensor
input_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0])

# Use the Cython function to apply the sin function
output_tensor = torch_extension.apply_sin(input_tensor)

print(output_tensor)
