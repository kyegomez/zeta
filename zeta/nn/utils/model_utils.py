import torch
import torch.nn as nn
from accelerate import Accelerator
from einops import rearrange

from zeta.nn.utils.helpers import exists



def print_num_params(model, accelerator: Accelerator):
    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of parameters in model: {n_params}")



class Block(nn.Module):
    def __init__(self,
                 dim,
                 dim_out,
                 groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
    
    def forward(self, 
                x,
                scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + 1

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self,
                 dim,
                 dim_out,
                 *,
                 time_emb_dim=None,
                 groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self,
                x,
                time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time_emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        
        return h + self.res_conv(x)
    
def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location=torch.device('cpu'))
    
