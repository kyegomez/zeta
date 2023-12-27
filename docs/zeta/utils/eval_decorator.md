# eval_decorator

# eval_decorator

## Summary:
This is a decorator function named **eval_decorator** from the utility package. It is used to ensure the automatic mode switching in pytorch's torch.nn.Module between evaluation (eval) and training (train) mode. 

When a method is wrapped with the **eval_decorator**, before invoking the method, the initial state of the model will be stored, and temporarily switch the model to evaluation state. The method then get executed. After execution, based on the previously saved state, the model would be reverted back to its original state (whether training or evaluation).

The primary purpose of this is to automate the switching back and forth between train and eval mode for a model during the running of a function which needs to be specifically run in eval mode.

## Code Explanation:
```python
def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner```

The **eval_decorator** takes a function as an argument, which needs to be wrapped to ensure the functionality as explained above. Here, 'fn' is the function to be wrapped.

The decorator function, **eval_decorator**, is defining another function, **inner**, inside it. **inner** function does the following:
- Stores the current state of the model (whether it is training or eval) in a variable was_training.
- Sets the model to eval mode using `self.eval()`.
- Calls the original function (to be wrapped), fn, with its arguments and keeps its return value in variable `out`.
- Sets back the model in the original state (which was stored in `was_training`).
- Returns `out`, output of the wrapped function.

## Parameters:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| fn |  function  |  The function to be decorated and thus wrapped inside the eval_decorator.  |

## Returns:

- Function `inner`: The evaluator function which is the wrapped version of the original function, fn.

## Example and Usage:

```python
import torch
import torch.nn as nn

# A demonstration model for example
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 10)
    
    @eval_decorator
