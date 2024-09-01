from torch import nn, Tensor
from zeta.nn.modules.feedforward import FeedForward


class TextHawkQueryProposal(nn.Module):
    """
    A module that represents the TextHawk query proposal model.

    Args:
        dim (int): The input and output dimension of the model.

    Attributes:
        dim (int): The input and output dimension of the model.
        ffn (FeedForward): The feed-forward network used in the model.

    """

    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.dim = dim

        self.ffn = FeedForward(dim, dim, 4, post_act_ln=True, swish=True)

    def forward(self, x: Tensor):
        x = self.ffn(x)

        # Maxpool
        maxpooled = nn.MaxPool1d(2, stride=2)(x)
        # print(maxpooled.shape)
        b, s, d = maxpooled.shape

        # Projection
        return nn.Linear(d, d)(maxpooled)


# x = torch.randn(1, 10, 512)
# model = TextHawkQueryProposal(512)
# output = model(x)
# print(output.shape)
