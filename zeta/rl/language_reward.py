import torch
from torch import nn
from torch.nn.modules.activation import Sigmoid


class LanguageReward(nn.Module):
    """
    Language Reward

    Args:
        ltype (str): Type of language reward.
            Options: ['cosine', 'l2', 'l1', 'bce']
        im_dim (int): Dimension of image embedding
        hidden_dim (int): Dimension of hidden layer
        lang_dim (int): Dimension of language embedding
        simfunc (torch.nn.Module): Similarity function


    Returns:
        reward (torch.Tensor): Reward for the given language embedding


    Examples:
    >>> import torch
    >>> from zeta.nn.modules.r3m import LanguageReward
    >>> x = torch.randn(1, 512)
    >>> y = torch.randn(1, 512)
    >>> z = torch.randn(1, 512)
    >>> lr = LanguageReward("cosine", 512, 512, 512)
    >>> print(lr(x, y, z))
    """

    def __init__(self, ltype, im_dim, hidden_dim, lang_dim, simfunc=None):
        super().__init__()
        self.ltype = ltype
        self.sim = simfunc
        self.sign = Sigmoid()
        self.pred = nn.Sequential(
            nn.Linear(im_dim * 2 + lang_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, e0, eg, le):
        """
        Forward pass for the language reward

        Args:
            e0 (torch.Tensor): Image embedding
            eg (torch.Tensor): Image embedding
            le (torch.Tensor): Language embedding

        Returns:
            reward (torch.Tensor): Reward for the given language embedding

        """
        info = {}
        return self.pred(torch.cat([e0, eg, le], -1)).squeeze(), info


# x = torch.randn(1, 512)
# y = torch.randn(1, 512)
# z = torch.randn(1, 512)

# lr = LanguageReward("cosine", 512, 512, 512)
# print(lr(x, y, z))
