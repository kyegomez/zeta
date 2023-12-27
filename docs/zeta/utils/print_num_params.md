# print_num_params

# Zeta Utils Documentation

## Class: print_num_params

Functionality:
The function 'print_num_params' prints the total number of trainable parameters of a given model. Model parameters are the attributes of the model that the algorithm modifies to enable the model to improve and adjust to the data better. Therefore, this function is important in determining the complexity of the model. More parameters in a model mean more complexity.

Typically higher parameter models have more training data and are better equipped to represent complex data patterns. However, having too many parameters can also lead to overfitting: the model might become too well adjusted to the training data and perform poorly on unseen data (high variance).

This function also checks if the PyTorch distributed package 'dist' is available and, if it is, prints the number of parameters on rank '0'. Rank in PyTorch's distributed package specifies the process rank (ID) for each process group. In a distributed environment (multiple GPUs), the function print_num_params will print the number of parameters from one GPU identified as rank '0'.

Here is the code definition:

```Python
def print_num_params(model):
    """
    Function to print out the number of trainable parameters in a PyTorch Model Model.

    Args:
        model (:obj: `torch.nn.Module`): The PyTorch Model.

    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if dist.is_available():
        if dist.get_rank() == 0:
            print(f"Number of parameters in model: {n_params}")
    else:
        print(f"Number of parameters in model: {n_params}")
```

Parameters:

| Parameter | Data Type | Description | Default Value |
| :--- | :--- | :--- | :--- |
| model | torch.nn.Module | The PyTorch model for which the number of parameters is to be calculated and printed. | - |

Other Functions Used:

- model.parameters(): Retrieves the model's parameters.
- p.requires_grad: Checks if the parameters require gradients (is trainable).
- p.numel(): Returns the total number of elements in the input tensor.
- dist.is_available(): Determines if PyTorch distributed is available.
- dist.get_rank(): Retrieves the rank in the current distributed group.

Here is an example of how to use this function.

```Python
import torch 
import torch.nn as nn
from torch import dist
from zeta.utils import print_num_params

model = nn.Linear(10,2) # A simple linear model

print_num_params(model)
```

Please note that if you are using this function in a distributed environment, you must first initialize your distributed environment correctly.

```Python
import torch 
import torch.nn as nn
from torch import dist
from zeta.utils import print_num_params

# initialize your distributed environment
dist.init_process_group(backend='nccl')

model = nn.Linear(10,2) # A simple linear model

print_num_params(model)
```

By using the function 'print_num_params', you can print out the total number of trainable parameters in your PyTorch models, which can have a significant impact on your model's complexity and its eventual performance.

Please note that this function works solely in a PyTorch environment and may not work with models built from other machine learning packages like Keras, TensorFlow, etc. It is also reliant on the dist package of PyTorch for distributed computations. This means you need to initialize your distributed environment if you are working with multiple GPUs.

Also, if you have specified some of the parameters of your model as non-trainable (by setting `requires_grad = False`), this function will not account for them.

## References & Resources
1. [Understanding Model Complexity](https://towardsdatascience.com/understanding-model-complexity-in-machine-learning-c5da3cc472f1)
2. [torch.numel()](https://pytorch.org/docs/stable/generated/torch.numel.html)
3. [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
4. [torch.distributed](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
