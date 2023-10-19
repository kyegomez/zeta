import torch
from torch import nn


class SubLN(nn.Module):
    """
    SubLN (Subtraction & Layer Normalization) module.

    This module computes the subln function:
    subln(x) = x + fout(LN(fin(LN(x))))

    Parameters:
    -----------
    d_model: int
        The number of expected features in the input x
    γ: float, optional
        Gain value for weight initialization. Default is 1.0.


    # Example usage

    # Usage example:
    import torch
    from zeta.nn.modules import SubLN

    model = SubLN(d_model=512)
    x = torch.randn(10, 512)
    out = model(x)
    print(out)

    """

    def __init__(self, d_model, γ=1.0):
        super(SubLN, self).__init__()

        # Define necessary layers and operations
        self.LN1 = nn.LayerNorm(d_model)
        self.fin = nn.Linear(d_model, d_model)  # Example layer for fin
        self.fout = nn.Linear(d_model, d_model)  # Example layer for fout
        self.LN2 = nn.LayerNorm(d_model)

        # Weight initialization
        self._initialize_weights(γ)

    def forward(self, x):
        """
        Forward pass for the subln function.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch_size, d_model]

        Returns:
        --------
        torch.Tensor
            Output tensor of shape [batch_size, d_model]

        """
        return x + self.fout(self.LN2(self.fin(self.LN1(x))))

    def _initialize_weights(self, γ):
        """
        Initialize weights of the module.

        Parameters:
        -----------
        γ : float
            Gain value for weight initialization.

        """
        for name, param in self.named_parameters():
            if "weight" in name:
                if name in ["fin.weight", "fout.weight", "out_proj.weight"]:
                    nn.init.xavier_normal_(param, gain=γ)
                elif name in ["q_proj.weight", "k_proj.weight"]:
                    nn.init.xavier_normal_(param, gain=1)
            elif "bias" in name:
                nn.init.zeros_(param)
