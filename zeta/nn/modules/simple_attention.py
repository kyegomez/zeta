import torch
import torch.nn.functional as F


def simple_attention(K, V, Q):
    _, n_channels, _ = K.shape
    A = torch.einsum("bct,bc1->bt1", [K, Q])
    A = F.softmax(A * n_channels ** (-0.5), 1)
    R = torch.einsum("bct, bt1->bc1", [V, A])
    return torch.cat((R, Q), dim=1)
