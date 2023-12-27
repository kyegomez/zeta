import torch
from zeta.nn.modules.simple_resblock import SimpleResBlock


def test_simple_resblock():
    # Initialize a SimpleResBlock with 10 channels
    resblock = SimpleResBlock(10)

    # Create a tensor of shape (1, 10)
    x = torch.rand(1, 10)

    # Pass the tensor through the SimpleResBlock
    output = resblock(x)

    # Check that the output has the same shape as the input
    assert output.shape == x.shape

    # Check that the output is not the same as the input
    # This checks that the SimpleResBlock is doing something to the input
    assert not torch.all(torch.eq(output, x))

    # Check that the output is a tensor
    assert isinstance(output, torch.Tensor)
