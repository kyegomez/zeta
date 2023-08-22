# FlashAttention Module
=====================

## Architecture
------------

The `FlashAttention` module is a part of the Zeta package and is designed to perform attention computation. It supports both regular attention and flash attention. The module takes query, key, and value tensors as input and computes the attention-weighted sum of the value tensors based on the similarity between the query and key tensors.

## Purpose
-------

The purpose of the `FlashAttention` module is to provide an efficient and flexible way to compute attention in neural networks. It allows users to choose between regular attention and flash attention based on their specific requirements. Flash attention is a variant of attention that leverages efficient GPU kernels to accelerate the attention computation process.

## Arguments
---------

The `FlashAttention` module accepts the following arguments:

-   `causal` (bool, default=False): Whether to apply causal masking during attention computation.
-   `dropout` (float, default=0.): Dropout probability.
-   `flash` (bool, default=True): Whether to use flash attention.

## Usage Examples
--------------

Here are three examples demonstrating different ways to use the `FlashAttention` module:

### Example 1: Regular Attention

```
import torch
from zeta import FlashAttention

q = torch.randn(2, 4, 6, 8)
k = torch.randn(2, 4, 10, 8)
v = torch.randn(2, 4, 10, 8)

attention = FlashAttention(causal=False, dropout=0.1, flash=False)
output = attention(q, k, v)

print(output.shape)  # torch.Size([2, 4, 6, 8])
```


In this example, we create random input tensors `q`, `k`, and `v` with appropriate shapes. We then create an instance of `FlashAttention` with causal masking disabled, dropout probability of `0.1`, and flash attention disabled. Finally, we pass the input tensors to the `FlashAttention` instance to compute the attention-weighted sum.

### Example 2: Flash Attention

```
import torch
from zeta import FlashAttention

q = torch.randn(2, 4, 6, 8)
k = torch.randn(2, 4, 10, 8)
v = torch.randn(2, 4, 10, 8)

attention = FlashAttention(causal=True, dropout=0.1, flash=True)
output = attention(q, k, v)

print(output.shape)  # torch.Size([2, 4, 6, 8])
```


In this example, we perform the same computation as in Example 1, but with flash attention enabled. Flash attention leverages efficient GPU kernels to accelerate the attention computation process, providing faster computation times compared to regular attention.

### Example 3: Masked Attention

```
import torch
from zeta import FlashAttention

q = torch.randn(2, 4, 6, 8)
k = torch.randn(2, 4, 10, 8)
v = torch.randn(2, 4, 10, 8)
mask = torch.ones(2, 4, 6, 10).bool()

attention = FlashAttention(causal=False, dropout=0.1, flash=True)
output = attention(q, k, v, mask=mask)

print(output.shape)  # torch.Size([2, 4, 6, 8])
```

In this example, we compute masked attention by providing a mask tensor to the `FlashAttention` module. The mask tensor has the same shape as the attention tensor and is used to mask out certain elements during the attention computation. This is useful when dealing with sequences of varying lengths or when certain elements should be ignored during the attention computation.