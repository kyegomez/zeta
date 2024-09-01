import torch
from einops import rearrange
from torch import Tensor


def expand(tensor: Tensor, pattern: str, **new_dims):
    """
    Reshape a tensor according to a specified pattern and new dimensions.

    Args:
        tensor (torch.Tensor): The input tensor to reshape.
        pattern (str): The pattern string defining the reshaping operation.
                       The pattern format follows 'input_pattern -> output_pattern',
                       where dimensions to combine or expand are placed in parentheses
                       and separated by whitespace on the input side, and directly
                       specified on the output side.
        **new_dims (dict): A dictionary where keys are dimension names in the output pattern,
                           and values are the sizes for these dimensions.

    Returns:
        torch.Tensor: The reshaped tensor according to the specified pattern and sizes.
    """

    # Validate the pattern format
    if "->" not in pattern:
        raise ValueError(
            "Pattern must contain '->' to separate input and output patterns."
        )

    input_pattern, output_pattern = pattern.split("->")
    input_pattern = input_pattern.strip()
    output_pattern = output_pattern.strip()

    # Prepare the dictionary for einops.rearrange by combining new_dims with input tensor's shape
    combined_dims = {
        **new_dims,
        **dict(zip(input_pattern.split(), tensor.shape)),
    }

    # Use einops.rearrange with the combined dimensions to perform the reshape
    reshaped_tensor = rearrange(
        tensor, f"{input_pattern} -> {output_pattern}", **combined_dims
    )

    return reshaped_tensor


# Example usage
if __name__ == "__main__":
    # Create a dummy tensor of shape [2, 50, 64] (for example, [Batch, Sequence, Features])
    tensor = torch.randn(2, 50, 64)

    # We want to reshape it to [2, 4, 25, 32], which could represent [Batch, Channels, Height, Width]
    pattern = "b (c h) (w f) -> b c h w"
    new_shape = expand(tensor, pattern, c=4, h=25, w=8, f=8)

    print(f"Original shape: {tensor.shape}")
    print(f"New shape: {new_shape.shape}")
