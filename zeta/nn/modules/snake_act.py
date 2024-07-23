import torch
import torch.nn as nn


class Snake(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(Snake, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return x + (1 / self.alpha) * torch.sin(self.alpha * x) ** 2


# # Example usage
# snake = Snake()
# x = torch.randn(10, 100, 100)  # Example input tensor
# output = snake(x)
# print(output)
