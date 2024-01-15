# Module/Function Name: MoERouter
    
class zeta.nn.modules.MoERouter(dim: int, num_experts: int, hidden_layers: int = None, mechanism: "str" = "softmax"):

Creates a module for routing input data to multiple experts based on a specified mechanism.

Args:
| Argument | Description                                  |
| -------- | -------------------------------------------- |
| dim      | The input dimension.                         |
| num_experts | The number of experts to route the data to. |
| hidden_layers | The number of hidden layers in the routing network. Defaults to None. |
| mechanism | The routing mechanism to use. Must be one of "softmax" or "gumbel". Defaults to "softmax". |

Raises:
ValueError: If the mechanism is not "softmax" or "gumbel".

Input Shape:
(B, SEQ_LEN, DIM) where SEQ_LEN is the sequence length and DIM is the input dimension.

Output Shape:
(B, SEQ_LEN, NUM_EXPERTS) where NUM_EXPERTS is the number of experts.

# Usage example:

x = torch.randn(2, 4, 6)
router = zeta.nn.modules.MoERouter(dim=6, num_experts=2, hidden_layers=[32, 64])
output = router(x)

# Note:
The above code demonstrates the use of the MoERouter module. It creates an instance of the MoERouter module with the input dimension of 6, routing the input data to 2 experts using a hidden layer configuration of [32, 64], and applies the module to the input tensor x.


# Introduction:
The MoERouter class is a module designed to route input data to multiple experts using a specified mechanism. It takes in input dimension, number of experts, hidden layers in the routing network, and routing mechanism as its arguments.

The MoERouter class acts as a flexible routing mechanism for distributing input data to multiple experts in a modular and configurable manner, allowing for different routing mechanisms to be applied based on the application requirements.

Note: The MoERouter class provides the flexibility to incorporate various routing mechanisms such as "softmax" and "gumbel", and supports the customization of the routing network with hidden layers. This enables the user to tailor the routing mechanism and configuration based on the specific use case and application scenarios.

For more details on the implementation and usage of the MoERouter class, refer to the provided documentation, examples, and usage guidelines.
