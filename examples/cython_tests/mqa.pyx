import torch
from torch import nn
cimport cython

cdef class MultiQueryAttention:
    cdef int embed_dim
    cdef int num_heads
    cdef int head_dim
    cdef object query_proj  # Treat nn.Linear as a Python object
    cdef object key_proj    # Treat nn.Linear as a Python object
    cdef object value_proj  # Treat nn.Linear as a Python object
    cdef object out_proj    # Treat nn.Linear as a Python object
    
    def __cinit__(self, int embed_dim, int num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Initialize nn.Linear layers as regular Python objects
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, self.head_dim)
        self.value_proj = nn.Linear(embed_dim, self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def forward(self, query, key, value):
        cdef int batch_size, seq_len, _

        # Assuming the input tensors are torch.Tensor objects
        batch_size, seq_len, _ = query.size()

        # Linear projections
        queries = self.query_proj(query)
        keys = self.key_proj(key)
        values = self.value_proj(value)

        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        values = values.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, values)

        # Concatenate and project the output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        return output
