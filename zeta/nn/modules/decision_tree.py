import torch
from torch import nn
import torch.nn.functional as F


class SimpleDecisionTree(nn.Module):
    """
    Simple decision tree model with residual connections and multi head output.


    Args:
        input_size (int): Input size of the model
        output_size (int): Output size of the model
        depth (int): Number of residual blocks
        heads (int): Number of output heads

    Example:
        >>> model = SimpleDecisionTree(
                input_size=10,
                output_size=5,
                depth=4,
                heads=3
            )
        >>> x = torch.randn(4, 10)
        >>> output = model(x)
        >>> print(output)
        [tensor([[-0.1015, -0.0114,  0.0370,  0.1362,  0.0436],
                 [-0.1015, -0.0114,  0.0370,  0.1362,  0.0436],
                 [-0.1015, -0.0114,  0.0370,  0.1362,  0.0436],
                 [-0.1015, -0.0114,  0.0370,  0.1362,  0.0436]],
                grad_fn=<AddmmBackward>), tensor([[-0.1015, -0.0114,  0.0370,  0.1362,  0.0436],
                 [-0.1015, -0.0114,  0.0370,  0.1362,  0.0436],
                 [-0.1015, -0.0114,  0.0370,  0.1362,  0.0436],
                 [-0.1015, -0.0114,  0.0370,  0.1362,  0.0436]],
                grad_fn=<AddmmBackward>), tensor([[-0.1015, -0.0114,  0.0370,  0.1362,  0.0436],
                 [-0.1015, -0.0114,  0.0370,  0.1362,  0.0436],
                 [-0.1015, -0.0114,  0.0370,  0.1362,  0.0436],
                 [-0.1015, -0.0114,  0.0370,  0.1362,  0.0436]],
                grad_fn=<AddmmBackward>)]
    """

    def __init__(
        self, input_size: int, output_size: int, depth: int, heads: int
    ):
        super(SimpleDecisionTree, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.depth = depth
        self.heads = heads

        # Initial input layer
        self.input_layer = nn.Linear(input_size, input_size)

        # Residual blocks with batch norm and dropout
        self.residual_blocks = nn.ModuleList([])
        for _ in range(depth):
            layers = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.BatchNorm1d(input_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(input_size, input_size),
                nn.BatchNorm1d(input_size),
                nn.ReLU(),
            )
            self.residual_blocks.append(layers)

        # Recurrent layer for temproal dynamics
        self.recurrent_layer = nn.LSTM(input_size, input_size, batch_first=True)

        # Multi head output system
        self.output_heads = nn.ModuleList(
            [nn.Linear(input_size, output_size) for _ in range(heads)]
        )

    def forward(self, x: torch.Tensor):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        x = self.input_layer(x)

        # Applying residual connections
        for block in self.residual_blocks:
            residual = x
            x = block(x) + residual

        # Recurrent layer
        x, _ = self.recurrent_layer(x.unsqueeze(0))
        x = x.squeeze(0)

        # Multi head output
        outputs = [head(x) for head in self.output_heads]
        return outputs


# # Params
# input_size = 10
# output_size = 5
# depth = 4
# heads = 3
# batch_size = 4

# # model
# model = SimpleDecisionTree(
#     input_size,
#     output_size,
#     depth,
#     heads
# )

# x = torch.randn(batch_size, input_size)

# output = model(x)
# print(output)
