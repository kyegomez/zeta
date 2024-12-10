
# SSM (Selective Scanning Module) Documentation

## Overview

The SSM (Selective Scanning Module) is a PyTorch-based module designed for selective scanning of input data. It is used to process input tensors by selectively extracting relevant information based on learned parameters. This documentation provides a comprehensive guide to understand, use, and maximize the functionality of the SSM module when imported from the `zeta.nn` library.


## Class Definition

### `SSM` Class

#### Constructor Parameters

- `in_features` (int): Size of the input features.
- `dt_rank` (int): Rank of the dt projection.
- `dim_inner` (int): Inner dimension of the dt projection.
- `d_state` (int): Dimension of the state.

### Methods

#### `forward` Method

#### Method Parameters

- `x` (torch.Tensor): Input tensor.
- `pscan` (bool, optional): Whether to use selective_scan or selective_scan_seq. (default: True)

## Functionality and Usage

The SSM module is designed to selectively scan input data using learned parameters. Here's how it works:

1. **Initialization**: The `SSM` class is initialized with parameters like `in_features`, `dt_rank`, `dim_inner`, and `d_state`.

2. **Forward Pass**: The `forward` method performs the core operation of selective scanning.

3. **Selective Scanning Modes**: The `pscan` parameter determines whether to use `selective_scan` or `selective_scan_seq` for the scanning process.

### Example Usage

Here are multiple usage examples of the SSM module importing it from the `zeta.nn` library:

```python
import torch

# Import SSM from zeta.nn
from zeta.nn import SSM

# Example 1: Creating an SSM instance
ssm = SSM(in_features=128, dt_rank=16, dim_inner=32, d_state=64)

# Example 2: Forward pass with selective_scan
output = ssm(torch.randn(10, 128))  # Output tensor after selective scanning

# Example 3: Forward pass with selective_scan_seq
output_seq = ssm(torch.randn(10, 128), pscan=False)  # Output using selective_scan_seq
```

## Additional Information

- The SSM module is designed to enhance the selective extraction of information from input data.
- You can customize its behavior by adjusting parameters during initialization.
- If you need to perform selective scanning in a sequential manner, set `pscan` to `False` in the `forward` method.

For more details and advanced usage, refer to the official PyTorch documentation and relevant research papers.

## References and Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [Research Paper: Selective Scanning Networks](https://example.com/research-paper)