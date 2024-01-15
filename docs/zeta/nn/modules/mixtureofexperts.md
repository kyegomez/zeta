
# Class Name: MixtureOfExperts

Mixture of Experts model.

Args:
| Argument | Data Type | Default Value | Description |
| --- | --- | --- | --- |
| dim | int | N/A | Input dimension |
| num_experts | int | N/A | Number of experts in the mixture |
| hidden_layers | int, optional | None | Number of hidden layers in the experts |
| mechanism | str, optional | "softmax" | Routing mechanism for selecting experts |
| custom_feedforward | callable, optional | None | Custom feedforward function for the experts |
| ff_mult | int, optional | 4 | Multiplier for the hidden layer dimension in the experts |
| *args | Variable length | N/A | Variable length argument list |
| **kwargs | Dict | N/A | Arbitrary keyword arguments |

Examples:
```python
import torch
from zeta.nn import MixtureOfExperts

x = torch.randn(2, 4, 6)
model = MixtureOfExperts(dim=6, num_experts=2, hidden_layers=[32, 64])
output = model(x)
print(output.shape)
```