import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalCrossAttention(nn.Module):
    """
    Multi-modal cross attention module for multi-modal (text and image) attention.

    Architecture
    ------------
    Timg -> Tllm
    Tllm -> Timg

    Args:
    - dim (int): Hidden dimension of the input
    - num_heads (int): Number of heads for multi-head attention
    - dropout (float): Dropout probability
    - qk_norm (bool): Whether to normalize the query and key vectors before computing attention weights

    Methods:
    - forward(Hllm, Himg): Forward pass of the cross attention module


    Usage
    -----
    from cross_attn.main import MultiModalCrossAttention

    dim = 512  # For example
    num_heads = 8
    cross_attn = MultiModalCrossAttention(dim, num_heads)
    Hllm_sample = torch.randn(32, 512, dim)  # Batch size = 32, Sequence length = 10
    Himg_sample = torch.randn(32, 512, dim)
    output = cross_attn(Hllm_sample, Himg_sample)
    print(output)

    print(output.shape)  # Expected: [32, 10, 512]
    """

    def __init__(self, dim, num_heads, dropout: int = 0.3, qk_norm: bool = True):
        super(MultiModalCrossAttention, self).__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.dk = dim // num_heads
        self.qk_norm = qk_norm

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Query, Key, Value projection layers for Timg -> Tllm
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)

        # Query, Key, Value projection layers for Tllm -> Timg (reverse)
        self.Wq_reverse = nn.Linear(dim, dim)
        self.Wk_reverse = nn.Linear(dim, dim)
        self.Wv_reverse = nn.Linear(dim, dim)

        # Output linear layer after attention computation
        self.linear_out = nn.Linear(2 * dim, dim)

    def forward(self, Hllm, Himg):
        """
        Hllm: Hidden states from Tllm
        Himg: Hidden states from Timg
        """

        # Timg -> Tllm
        Qcross = self.Wq(Hllm)
        Kcross = self.Wk(Himg)
        Vcross = self.Wv(Himg)

        if self.qk_norm:
            # Normalize Qcross and Kcross
            Qcross = self.norm(Qcross)
            Kcross = self.norm(Kcross)
        else:
            pass

        # Compute attention weights, why is Kcross being transposed?
        # Because we want to multiply the query with the key, and the key has to be transposed
        # Original code
        # attn_weights = F.softmax(Qcross @ Kcross.transpose(-2, -1) / torch.sqrt(torch.tensor(self.dk).float()), dim=-1)

        # New code
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            # attention, should Kcross be tranposed here?
            attn_weights = F.scaled_dot_product_attention(Qcross, Kcross, Vcross)

            # dropout
            attn_weights = self.dropout(attn_weights)

            # rearrange to original shape
            # attn_weights = rearrange(out, 'b h n d -> b n (h d)'

        print(
            f"attn_weights shape: {attn_weights.shape}, and vcross shape:"
            f" {Vcross.shape}"
        )

        # what does the @ symbol mean?
        # It's matrix multiplication
        # https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication
        # Hcross = attn_weights @ Vcross
        # New code
        # Hcross = attn_weights + Vcross
        # newest code
        Hcross = torch.matmul(attn_weights, Vcross)

        # model 2
        # -----------------------

        # Tllm -> Timg (Symmetric process)
        Qcross_reverse = self.Wq_reverse(Himg)
        Kcross_reverse = self.Wk_reverse(Hllm)
        Vcross_reverse = self.Wv_reverse(Hllm)

        # attn_weights_reverse = F.softmax(Qcross_reverse @ Kcross_reverse.transpose(-2, -1) / torch.sqrt(torch.tensor(self.dk).float()), dim=-1)
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            # attention, should Kcross be tranposed here?
            attn_weights_reverse = F.scaled_dot_product_attention(
                Qcross_reverse, Kcross_reverse, Vcross_reverse
            )

            # dropout
            attn_weights_reverse = self.dropout(attn_weights_reverse)

            # rearrange to original shape
            # attn_weights_reverse = rearrange(out, 'b h n d -> b n (h d)')

        # old code
        # Hcross_reverse = attn_weights_reverse @ Vcross_reverse
        # new code
        # Hcross_reverse = attn_weights_reverse + Vcross_reverse
        # newest code
        Hcross_reverse = torch.matmul(attn_weights_reverse, Vcross_reverse)

        # Concatenate the results
        output = torch.cat((Hcross, Hcross_reverse), dim=-1)

        # Pass through linear layer
        output = self.linear_out(output)

        return output
