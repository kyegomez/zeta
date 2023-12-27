# print_num_params

# Module Name: utils.print_num_params

## Function: 
```python
def print_num_params(model):
```
This function calculates the total number of trainable parameters in a PyTorch model and prints this number. This is a utility function that can be used to monitor the complexity of the model.

## Arguments:

| Argument | Type | Description |
| --- | --- | --- |
| model | `torch.nn.Module` | The model for which you want to count the number of parameters. |


## Function Body:

This function loops over all the parameters of the model that require gradient computation (i.e., trainable parameters), counts their number (numel), and sums them up to get the total count of parameters.

In a distributed training setup, the function checks whether the distributed communication package (`dist`) is available. If it is, only the specified process (the one with rank 0), prints the number of parameters. If the distributed communication package is not available (which means it's not a distributed setup), the function just prints the number of parameters in the model.

## Usage Example:

```python
import torch
import torch.nn as nn
from zeta.utils import print_num_params

# Define a simple model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

# Initialize the model
model = Model()
# Print the number of parameters in the model
print_num_params(model)
```

In the above example, the Model has a single linear layer with an input feature size of 4 and an output feature size of 2. So, the number of parameters in this model will be `(4 * 2) + 2 = 10`, where 4 and 2 are weight parameters for each input and output features and added two because of the bias parameters for the outputs.

Running the `print_num_params` on this `model` will output:

```
Number of parameters in model: 10
```

## Notes:

1. This function counts only the parameters that are trainable i.e., require gradient computation. If your model has layers or parameters with `requires_grad` set to False, those will not be counted.

2. In case of distributed training, `dist.is_available()` is used to determine whether the distributed communication package is available.

3. If the
