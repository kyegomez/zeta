# eval_decorator

# Module Name: `eval_decorator`

**Note:** The following is a simplified illustrative example of the `eval_decorator` function.

`eval_decorator` is a higher-order function that takes another function as a parameter and wraps it, providing additional functionality. It is a decorator specifically built for Torch's `nn.Module` objects, ensuring the wrapped method switches to evaluation mode (`.eval()`) before execution and restores the model's original mode (training or evaluation) afterwards.

## Function Declaration
```python
def eval_decorator(fn):
    """
    Decorator to ensure a method switches to eval mode before execution
    and returns to its original mode afterwards. For torch.nn.Module objects.

    Args:
        fn (function): The function to wrap.

    Returns:
        function: The wrapped function.
    """

    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner
```

## Parameters

Parameter | Type | Default | Description
--- | --- | --- | ---
`fn` | `function` | None | The function or method to be wrapped by `eval_decorator`.

## Return Type
**Type:** `function` (The wrapped function)

## How it Works

The `eval_decorator` function wraps around another function, `fn` and adds some extra steps before and after it runs. Inside, it defines another function named `inner`. This `inner` function does the following:

1. Captures the original training state (True or False) of the `nn.Module` object before it is executed.

2. Switches the module to evaluation mode by invoking `self.eval()`. (Note: `self` refers to an instance of a class that inherits from `torch.nn.Module`.)

3. Executes the wrapped function `fn`.

4. Restores the original training state.

5. Returns the output of the wrapped function `fn`.

In summary, `eval_decorator` is a decorator - a tool in Python for wrapping functions. It modifies the behavior of a function, providing a way to add features or characteristics, in this case handling the switch between training and evaluation mode in PyTorch.

## Usage Example 1
```python
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

    @eval_decorator
    def forward(self, x):
        x = self.conv1(x)
        return x


model = Net()
print(model.training)  # True - The model is initially in training mode

# Using the wrapped forward method switches to eval mode and back to training mode
output = model(torch.randn(1, 1, 64, 64))
print(model.training)  # True - Mode is restored back to original state
```
## Usage Example 2

Applying the decorator to a different method:
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

    def forward(self, x):
        x = self.conv1(x)
        return x

    @eval_decorator
    def predict(self, x):
        # This method uses the model in evaluation mode
        with torch.no_grad():
            return self.forward(x)


model = Net()
print(model.training)  # True

prediction = model.predict(torch.randn(1, 1, 64, 64))
print(model.training)  # Still True, as predict() method used eval_decorator
```

## Usage Example 3

Usage in a more complex module:
```python
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(...)

        self.classifier = nn.Linear(...)

    @eval_decorator
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = Classifier()
output = model(torch.randn(5, 3, 32, 32))
print(output)
```
In all these examples, any code section using `@eval_decorator` temporarily switches the mode of the model to evaluation mode, executes the decorated function, then restores the mode back to its original state.

## Tips

- Be careful not to use the decorator incorrectly. It should only be used on methods inside classes that are directly or indirectly subclassing `torch.nn.Module`.

- The decorator is useful when you want to ensure a function is always run in eval mode, without having
