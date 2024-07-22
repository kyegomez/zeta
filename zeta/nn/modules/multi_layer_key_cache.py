import torch
import torch.nn as nn


class MultiLayerKeyValueAttention(nn.Module):
    """
    Multi-layer key-value attention module.

    Args:
        embed_size (int): The size of the input embeddings.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of layers.
        kv_layers (int): The number of key-value layers.

    Attributes:
        num_heads (int): The number of attention heads.
        num_layers (int): The number of layers.
        kv_layers (int): The number of key-value layers.
        embed_size (int): The size of the input embeddings.
        head_dim (int): The dimension of each attention head.

        values (nn.ModuleList): List of value projection layers for each key-value layer.
        keys (nn.ModuleList): List of key projection layers for each key-value layer.
        queries (nn.ModuleList): List of query projection layers for each layer.
        fc_out (nn.Linear): Output linear layer.

    """

    def __init__(self, embed_size, num_heads, num_layers, kv_layers):
        super(MultiLayerKeyValueAttention, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kv_layers = kv_layers  # m in the description
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

        # Define the key and value projections for each layer
        self.values = nn.ModuleList(
            [
                nn.Linear(embed_size, embed_size, bias=False)
                for _ in range(kv_layers)
            ]
        )
        self.keys = nn.ModuleList(
            [
                nn.Linear(embed_size, embed_size, bias=False)
                for _ in range(kv_layers)
            ]
        )

        # Define the query projections for each layer
        self.queries = nn.ModuleList(
            [
                nn.Linear(embed_size, embed_size, bias=False)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries):
        """
        Forward pass of the multi-layer key-value attention module.

        Args:
            values (torch.Tensor): The values tensor of shape (N, value_len, embed_size).
            keys (torch.Tensor): The keys tensor of shape (N, key_len, embed_size).
            queries (torch.Tensor): The queries tensor of shape (N, query_len, embed_size).

        Returns:
            torch.Tensor: The output tensor of shape (N, query_len, embed_size).

        """
        N = queries.shape[0]
        value_len, key_len, query_len = (
            values.shape[1],
            keys.shape[1],
            queries.shape[1],
        )

        out = torch.zeros(N, query_len, self.embed_size).to(values.device)

        for layer in range(self.num_layers):
            kv_index = layer % self.kv_layers

            values_layer = self.values[kv_index](values).view(
                N, value_len, self.num_heads, self.head_dim
            )
            keys_layer = self.keys[kv_index](keys).view(
                N, key_len, self.num_heads, self.head_dim
            )
            queries_layer = self.queries[layer](queries).view(
                N, query_len, self.num_heads, self.head_dim
            )

            energy = torch.einsum(
                "nqhd,nkhd->nhqk", [queries_layer, keys_layer]
            )
            attention = torch.softmax(
                energy / (self.embed_size ** (1 / 2)), dim=3
            )
            out_layer = torch.einsum(
                "nhql,nlhd->nqhd", [attention, values_layer]
            ).reshape(N, query_len, self.embed_size)

            out += out_layer

        out = self.fc_out(out)
        return out


# # Example usage
# embed_size = 256
# num_heads = 8
# num_layers = 4
# kv_layers = 2  # Number of layers with their own KV heads

# mlkv_attention = MultiLayerKeyValueAttention(embed_size, num_heads, num_layers, kv_layers)
# values = torch.rand(32, 10, embed_size)  # batch size 32, sequence length 10
# keys = torch.rand(32, 10, embed_size)
# queries = torch.rand(32, 10, embed_size)

# output = mlkv_attention(values, keys, queries)
# print(output.shape)
