
from einops.layers.torch import Rearrange
from torch import nn


class PatchEmbeddings(nn.Module):
    def __init__(
            self, 
            dim_in, 
            dim_out, 
            seq_len
        ):
        super().__init__()
        self.embedding = nn.Sequential(
            Rearrange('... rd -> ... (r d)'),
            nn.LayerNorm(seq_len * dim_in),
            nn.Linear(seq_len * dim_in, dim_out),
            nn.LayerNorm(dim_out),
        )
    
    def forward(self, x):
        return self.embedding(x)