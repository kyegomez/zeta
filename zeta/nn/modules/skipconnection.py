import torch.nn as nn


class SkipConnection(nn.Module):
    """
    A helper class to implement skip connections.
    Adds two input tensors element-wise.

    # Example usage
    from zeta.nn import SkipConnection
    tensor1 = torch.randn(1, 1024, 512)
    tensor2 = torch.randn(1, 1024, 512)
    skip_connection = SkipConnection()
    output = skip_connection(tensor1, tensor2)
    print(output.shape)

    """

    def __init__(self):
        super(SkipConnection, self).__init__()

    def forward(self, tensor1, tensor2):
        """
        Forward pass to add two tensors.

        Args:
            tensor1 (torch.Tensor): The first tensor.
            tensor2 (torch.Tensor): The second tensor, which should have the same shape as tensor1.

        Returns:
            torch.Tensor: The element-wise sum of tensor1 and tensor2.
        """
        try:
            if tensor1.size() != tensor2.size():
                raise ValueError(
                    "The size of both tensors must be the same for element-wise"
                    " addition."
                )

            return tensor1 + tensor2
        except Exception as error:
            print(f"Error: {error}")
            raise error
