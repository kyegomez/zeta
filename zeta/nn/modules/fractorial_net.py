import torch
import torch.nn as nn
import torch.nn.functional as F


class FractorialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth: int = 3):
        super(FractorialBlock, self).__init__()
