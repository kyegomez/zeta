# `PositionInterpolationEmbeddings` Class
=====================================

The `PositionInterpolationEmbeddings` class implements positional embeddings that interpolate between sinusoidal and learned embeddings.

## Attributes
------
-   `inv_freq` (torch.Tensor): Cached inverse frequencies.
-   `max_seq_len_cached` (int): Maximum sequence length cached.
-   `scale` (float): Scale of the sinusoidal embedding.
-   `cos_cached` (torch.Tensor): Cached cosine values.
-   `sin_cached` (torch.Tensor): Cached sine values.

## Methods
-------

### `__init__(self, dim: int = None, max_positions: int = 2048, base: int = 10000, device = None)`

The constructor for the `PositionInterpolationEmbeddings` class. Initializes the inverse frequencies, the maximum sequence length cached, the scale, and the cached cosine and sine values.

#### Parameters

-   `dim` (int, optional): Dimension of the input embedding.
-   `max_positions` (int, optional): Maximum number of positions to embed. Default is 2048.
-   `base` (int, optional): Base of the sinusoidal embedding. Default is 10000.
-   `device` (torch.device, optional): Device to store the embeddings on.

#### Example

```
embeddings = PositionInterpolationEmbeddings(dim=512, max_positions=2048, base=10000, device=torch.device('cuda'))
```


### `forward(self, x, seq_len=None)`

Forward pass of the `PositionInterpolationEmbeddings`.

#### Parameters

-   `x` (torch.Tensor): Input tensor.
-   `seq_len` (int, optional): Sequence length.

#### Returns

-   `cos_cached` (torch.Tensor): Cached cosine values.
-   `sin_cached` (torch.Tensor): Cached sine values.

#### Example

```
cos_cached, sin_cached = embeddings.forward(x, seq_len=512)
```


## Usage Examples
--------------

### Example 1: Initialize PositionInterpolationEmbeddings

In this example, we will initialize `PositionInterpolationEmbeddings` with a dimension of 512, a maximum number of positions of 2048, a base of 10000, and a device of 'cuda'.

```python
embeddings = PositionInterpolationEmbeddings(dim=512, max_positions=2048, base=10000, device=torch.device('cuda'))
```


### Example 2: Forward Pass of PositionInterpolationEmbeddings

In this example, we will perform a forward pass of `PositionInterpolationEmbeddings` with an input tensor `x` and a sequence length of 512.

```python
x = torch.randn(1, 512, 512).to(torch.device('cuda'))
cos_cached, sin_cached = embeddings.forward(x, seq_len=512)
```


### Example 3: Using PositionInterpolationEmbeddings in a Model

In this example, we will use `PositionInterpolationEmbeddings` in a model.

```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embeddings = PositionInterpolationEmbeddings(dim=512, max_positions=2048, base=10000, device=torch.device('cuda'))

    def forward(self, x):
        cos_cached, sin_cached = self.embeddings(x, seq_len=x.size(1))
        return cos_cached, sin_cached

model = Model().to(torch.device('cuda'))
x = torch.randn(1, 512, 512).to(torch.device('cuda'))
cos_cached, sin_cached = model(x)
```
