from types import ModuleType
from typing import List
import importlib
import sys


class LazyLoader:
    """
    A class for lazy loading modules in __init__ files.
    """

    def __init__(self, module_name: str):
        """
        Initialize the LazyLoader.

        Args:
            module_name (str): The name of the module to be lazy loaded.
        """
        self._module_name = module_name
        self._module: ModuleType | None = None

    def __getattr__(self, name: str):
        """
        Lazily import the module and return the requested attribute.
        """
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return getattr(self._module, name)


def lazy_import(module_names: List[str]) -> None:
    """
    Set up lazy imports for the given module names.

    Args:
        module_names (List[str]): A list of module names to be lazy loaded.

    Example:
        lazy_import(['numpy', 'pandas'])
    """
    for module in module_names:
        setattr(
            sys.modules[__name__], module.split(".")[-1], LazyLoader(module)
        )
