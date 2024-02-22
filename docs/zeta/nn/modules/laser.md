# Module/Function Name: LayerSelectiveRankReduction

The `LayerSelectiveRankReduction` (LASER) module replaces specific weight matrices in a Transformer model by their low-rank approximations for both 2D and 3D tensors.

`LASER` is a pyTorch based module that aids in approximating weight matrices using a low rank matrix decomposition. Examples where the memory consumption footprint needs to be controlled and approximated to manage memory constraints. This module is particularly effective for text datasets which can require high computational resources.

The main attribute for `LASER` is `rank_fraction` which denotes the fraction of the maximum rank to reserve in the approximation, with the value ranging from 0 to 1.

**Example Usage:**

```python
import torch
from torch import nn

from zeta.nn import LASER

# Dimension of the weight matrix
weight_dim = 512

# Example weight matrix (2D tensor)
W_2d = torch.randn(weight_dim, weight_dim)

# Example weight batch (3D tensor)
W_3d = torch.randn(10, weight_dim, weight_dim)

# Fraction of the rank to preserve
rank_fraction = 0.9

# Create the LASER module
laser = LASER(rank_fraction)

# Apply LASER to 2D and 3D tensors to obtain low-rank approximations
W_2d_low_rank = laser(W_2d)
W_3d_low_rank = laser(W_3d)

# Output the shape of the approximated matrices
print(
    W_2d_low_rank.shape
)  # The shape of the approximated 2D matrix will be the same as the original matrix
print(
    W_3d_low_rank.shape
)  # The shape of the approximated matrices will be the same as the original 3D tensor
```

**Additional Tips:**

For better performance, it's recommended that developers monitor memory and resource usage while applying LASER for large matrices. Additionally, it is advised to adequately test the optimized model performance after using the `LASER` module to maintain required accuracy whilst significantly reducing memory usage.

**References and Resources:**

- [LASER PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.solve.html)

Further exploration of memory reduction techniques for large-scale optimized machine learning models can be referenced for a more in-depth understanding.

This is an example of a module that replaces specific weight matrices with their low-rank approximations. Developers can refer to this documentation as a reference and template to create a similar documentation for other modules or frameworks.
