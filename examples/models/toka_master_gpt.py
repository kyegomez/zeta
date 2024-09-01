import torch
from torch import nn, Tensor
import torch.nn.functional as F
from zeta.nn.attention.multiquery_attention import MultiQueryAttention
from zeta.nn import OutputHead


class TokaTransformerBlock(nn.Module):
    """
    Transformer block used in the Toka model.

    Args:
        dim (int): The input dimension.
        dim_head (int): The dimension of each attention head.
        heads (int): The number of attention heads.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        dropout (float, optional): The dropout rate. Defaults to 0.1.

    Attributes:
        dim (int): The input dimension.
        dim_head (int): The dimension of each attention head.
        heads (int): The number of attention heads.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        dropout (float): The dropout rate.
        attn (MultiQueryAttention): The multi-query attention module.
        mlp (nn.Sequential): The feed-forward network module.
        norm (nn.LayerNorm): The layer normalization module.

    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        ff_mult: int,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.ff_mult = ff_mult
        self.dropout = dropout

        # Attention
        self.attn = MultiQueryAttention(
            dim,
            heads,
        )

        # FFn
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.ELU(),
            nn.Linear(dim * ff_mult, dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

        # LayerNorm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the TokaTransformerBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        skip = x
        x, _, _ = self.attn(x)

        # Add with the skip connection
        x = x + skip
        x = self.norm(x)
        skip_two = x

        # MLP
        x = self.mlp(x)
        x = x + skip_two
        return self.norm(x)


class TokaTransformer(nn.Module):
    """
    A transformer model based on the Toka architecture.

    Args:
        dim (int): The dimension of the input and output tensors.
        dim_head (int): The dimension of each head in the multi-head attention mechanism.
        heads (int): The number of attention heads.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        depth (int, optional): The number of transformer layers. Defaults to 6.

    Attributes:
        dim (int): The dimension of the input and output tensors.
        dim_head (int): The dimension of each head in the multi-head attention mechanism.
        heads (int): The number of attention heads.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        dropout (float): The dropout probability.
        layers (nn.ModuleList): The list of transformer layers.
        norm (nn.LayerNorm): The layer normalization module.

    """

    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        depth: int = 6,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.ff_mult = ff_mult
        self.dropout = dropout

        # Transformer layer
        self.layers = nn.ModuleList(
            [
                TokaTransformerBlock(dim, dim_head, heads, ff_mult, dropout)
                for _ in range(depth)
            ]
        )

        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the TokaTransformer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        x = self.norm(x)

        for layer in self.layers:
            x = layer(x)

        return OutputHead(self.dim, 1)(x)


# x = torch.randn(1, 10, 512)
# model = TokaTransformer(512, 64, 8, 4)
# out = model(x)
# print(f"Transformer output shape: {out.shape}")
# print(f"Transformer output: {out}")


class TokaCriticNetworkBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_mult: int,
        dropout: float = 0.1,
        num_layers: int = 256,
        transformer: bool = False,
        transformer_depth: int = 6,
    ):
        """
        Initialize the TokaCriticNetworkBlock.

        Args:
            dim (int): The input dimension.
            ff_mult (int): The multiplier for the feed-forward layer dimension.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.dim = dim
        self.ff_mult = ff_mult
        self.dropout = dropout
        self.transformer = transformer

        self.act = nn.Tanh()

        self.lstm_head = nn.LSTM(
            dim, dim, num_layers=num_layers, dropout=dropout
        )
        self.transformer = TokaTransformer(
            dim,
            dropout=dropout,
            depth=transformer_depth,
        )

        # Sequential
        self.mlp_small = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.ELU(),
            nn.Linear(dim * ff_mult, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the TokaCriticNetworkBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        # B, S, D
        x = self.act(x)
        skip = x
        print(f"Skip shape: {skip.shape}")

        # LSTM
        if self.transformer is True:
            x = self.transformer(x)
        else:
            x, _ = self.lstm_head(x)

        print(x.shape)

        # Concatenate
        lstm_output = torch.cat((skip, x), dim=1)
        print(lstm_output.shape)

        # Apply the MLP to the lstm outpout
        x = self.mlp_small(lstm_output)

        return nn.Linear(self.dim, self.dim)(x)


# # Forward
# x = torch.randn(1, 10, 512)

# # Model
# model = TokaCriticNetworkBlock(512, 4)

# # Forward
# out = model(x)
# print(out)


"""
linear -> layernorm -> tanh -> 3 layer mlp using elu -> linaer 
-> mean of gaussian distribution, standard deviation of the the gaussian distribution
"""


class TokaPolicyBlock(nn.Module):
    """
    A class representing a policy block in the Toka model.

    Args:
        dim (int): The dimension of the input and output tensors. Default is 256.
        dropout (float): The dropout probability. Default is 0.1.
        ff_mult (int): The multiplier for the dimension of the hidden layer in the MLP. Default is 4.
        actions (int): The number of output actions. Default is 2.

    Attributes:
        dim (int): The dimension of the input and output tensors.
        dropout (float): The dropout probability.
       e ff_mult (int): The multiplier for the dimension of the hidden layer in the MLP.
        actions (int): The number of output actions.
        proj (nn.Linear): The linear projection layer.
        norm (nn.LayerNorm): The layer normalization layer.
        tanh (nn.Tanh): The hyperbolic tangent activation function.
        mlp (nn.Sequential): The multi-layer perceptron.
        soft (nn.Softplus): The softplus activation function.
        final_proj (nn.Linear): The final linear projection layer.

    Methods:
        forward(x: Tensor) -> Tensor:
            Performs the forward pass of the policy block.

    """

    def __init__(
        self,
        dim: int = 256,
        dropout: float = 0.1,
        ff_mult: int = 4,
        actions: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.actions = actions

        # Linear
        self.proj = nn.Linear(dim, dim)

        # LayerNorm
        self.norm = nn.LayerNorm(dim)

        # Tanh
        self.tanh = nn.Tanh()

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.ELU(),
            nn.Linear(dim * ff_mult, dim),
            nn.ELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

        # Softplus
        self.soft = nn.Softplus()

        # Final proj
        self.final_proj = nn.Linear(dim, actions)

        # Initialize weights using truncated normal distribution
        nn.init.trunc_normal_(self.proj.weight, std=1 / (dim**0.5))
        nn.init.trunc_normal_(self.mlp[0].weight, std=1 / (dim**0.5))
        nn.init.trunc_normal_(self.mlp[2].weight, std=1 / (dim**0.5))
        nn.init.trunc_normal_(self.mlp[4].weight, std=1 / (dim**0.5))
        nn.init.trunc_normal_(self.final_proj.weight, std=0.0001)

        # Initialize biases to zero
        self.proj.bias.data.zero_()
        self.mlp[0].bias.data.zero_()
        self.mlp[2].bias.data.zero_()
        self.mlp[4].bias.data.zero_()
        self.final_proj.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the policy block.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor containing the means and standard deviations of the actions.

        """
        x = self.proj(x)

        # Norm
        x = self.norm(x)

        # Tanh
        x = self.tanh(x)

        # MLP
        x = self.mlp(x)

        # Final linear
        x = self.proj(x)

        # Mean and log std
        means, log_std = x.chunk(2, dim=1)
        stds = F.softplus(log_std)

        # Return
        return means, stds


# x = torch.randn(1, 10, 512)
# model = TokaPolicyBlock(512)
# out = model(x)
# print(out)
