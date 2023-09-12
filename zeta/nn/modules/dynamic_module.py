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
    dynmaic_module.remove('relu')
    
    """
    def __init__(self):
        super(DynamicModule, self).__init__()
        self.module_dict = nn.ModuleDict()

    def add(self, name, module):
        """
        Add a module to the container

        Args:
            name (str) the name of the module
            module(nn.Module) the module to add
        """
        self.module_dict[name] = module

    def remove(self, name):
        """
        Remove a module from the container

        Args:
            name (str) the name of the module to remove
        """
        del self.module_dict[name]
    
    def forward(self, x):
        """
        Forward pass through the modules

        Args:
            x (Tensor) the input tensor
        
        Returns:
            Tensor: the output tensor
        
        """
        for module in self.module_dict.values():
            x = module(x)
        return x
    
