# FeedbackBlock

---

`FeedbackBlock` is a class that extends the `torch.nn.Module` class. As a crucial part of the neural network, this class perfectly illustrates the aspect of modularity that deep learning models can have.

`FeedbackBlock` is a namespace that hosts operations and behaves to transformations in such a way that all of its submodules follow along. Its main role is to handle the feedback connections in neural networks while wrapping another module. The feedback connection is a very common architecture in deep learning where the output from one layer is used as additional input to the same layer in subsequent passes.

## Class Definition:

```python
class FeedbackBlock(nn.Module):
```

The `FeedbackBlock` class has one primary attribute: `submodule`. The `submodule` argument represents the "submodule" of the current instance of the `FeedbackBlock` class. It is an instance of `torch.nn.Module`.

In the initial definition, `FeedbackBlock` takes a `submodule` as an argument and assigns it to an attribute of the class.

```python
def __init__(self, submodule):
    """
    Initializes the FeedbackBlock module.

    Args:
        submodule (nn.Module): The submodule to be used within the FeedbackBlock.
    """
    super().__init__()
    self.submodule = submodule
```
    
The `submodule` will be triggered during the forward pass of the `FeedbackBlock`, with the input subjected to the feedback mechanism.

_Note_: If another Module is assigned as an attribute to a Module, PyTorch will understand that it owns Parameters that can be part of the optimization problem.

## Forward Method:

```python
def forward(self, x: torch.Tensor, feedback, *args, **kwargs):
    """
    Performs a forward pass through the FeedbackBlock.

    Args:
        x (torch.Tensor): The input tensor.
        feedback: The feedback tensor.
        *args: Additional positional arguments to be passed to the submodule's forward method.
        **kwargs: Additional keyword arguments to be passed to the submodule's forward method.

    Returns:
        torch.Tensor: The output tensor after passing through the FeedbackBlock.
    """
    if feedback is not None:
        x = x + feedback
    return self.submodule(x, *args, **kwargs)
```

The `forward` method does the actual computation or transformation. First, the `feedback` tensor is checked. If it exists (if it's not None), it is added into the input tensor. Once the feedback has been integrated into the input, it calls the forward method of the submodule. Any additional arguments would be directly passed to the submodule's forward method. The output of the submodule's forward pass is the final output we return.

# Usage:

The usage of `FeedbackBlock` is essentially to encapsulate a module in a network that performs a feedback operation. Let's take a simple scenario where you have a neural network `model` with a linear layer `nn.Linear(10,10)`:

```python
import torch
import torch.nn as nn
from zeta.nn import FeedbackBlock
   

# Define a simple linear network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

# Instantiate the simple network
simple_net = SimpleNet()
   
# Wrapping the simple network with a FeedbackBlock
feedback_net = FeedbackBlock(simple_net)

# Usage in a training loop:
x = torch.rand((64, 10)) # Assume an input tensor for batch of 64.

# Initialize feedback
feedback = None

for _ in range(100): # 100 steps
    y = feedback_net(x, feedback)
    feedback = y.detach() # Detach() to avoid backpropagating gradients through time
    # ... Rest of training loop here
```

In the code above, the output from one pass will be fed back into the module during the next pass. This allows the network to adjust its weights accordingly, based on this continuous feedback loop it’s in.

Remember that whenever using the FeedbackBlock to encapsulate a network module, the forward method of the base module, must be designed to handle the feedback tensor that will be passed onto it.

In charging forward into more complex architectures with dynamic networks or feedback connections, `FeedbackBlock` will be of immense help, abstracting the complexities away from your specific model and keeping your code modular and easy to follow.
