import torch
from torch import nn


class DynamicModule(nn.Module):
    """
    A container that allows for dynamic addition, removal, and modification
    of modules

    examples
    ````
    dynamic_module = DynamicModule()
    dynamic_module.add('linear', nn.Linear(10, 10))
    dynamic_module.add('relu', nn.ReLU())
    output = dynamic_module(torch.randn(1, 10))
    dynamic_module.remove('relu')

    """

    def __init__(
        self,
        forward_method=None,
    ):
        super(DynamicModule, self).__init__()
        self.module_dict = nn.ModuleDict()
        self.forward_method = forward_method

    def add(self, name, module):
        """
        Add a module to the container

        Args:
            name (str) the name of the module
            module(nn.Module) the module to add
        """
        if isinstance(name, list):
            name = ".".join(name)
        if not isinstance(module, nn.Module):
            raise ValueError("Module must be a nn.Module")
        if name in self.module_dict:
            raise ValueError("Module name must be unique")
        self.module_dict[name] = module

    def remove(self, name):
        """
        Remove a module from the container

        Args:
            name (str) the name of the module to remove
        """
        if isinstance(name, list):
            name = ".".join(name, list)
        if name not in self.module_dict:
            raise ValueError("module name does not exist")
        del self.module_dict[name]

    def forward(self, x):
        """
        Forward pass through the modules

        Args:
            x (Tensor) the input tensor

        Returns:
            Tensor: the output tensor

        """
        if self.forward_method is not None:
            return self.forward_method(self.module_dict, x)
        for module in self.module_dict.values():
            x = module(x)
        return x

    def save_state(self, path):
        torch.save(self.state_dict(), path)

    def load_state(self, path):
        self.load_state_dict(torch.load(path))
