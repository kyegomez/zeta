from functools import partial, wraps
from torch import nn

def exists(val):
    """
    Check if the value is not None.
    
    Args:
        val: The value to check.
        
    Returns:
        bool: True if value exists (is not None), False otherwise.
    """
    return val is not None

def default(val, d):
    """
    Return the value if it exists, otherwise return a default value.

    Args:
        val: The value to check.
        d: The default value to return if val is None.

    Returns:
        The value if it exists, otherwise the default value.
    """
    return val if exists(val) else d

def once(fn):
    """
    Decorator to ensure the function is only called once.

    Args:
        fn (function): The function to wrap.

    Returns:
        function: The wrapped function.
    """
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner


print_once = once(print)

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


def cast_tuple(val, depth):
    """
    Cast a value to a tuple of a specific depth.

    Args:
        val: Value to be cast.
        depth (int): Depth of the tuple.

    Returns:
        tuple: Tuple of the given depth with repeated val.
    """
    return val if isinstance(val, tuple) else (val,) * depth


def maybe(fn):
    """
    Decorator that calls a function if the first argument exists.

    Args:
        fn (function): The function to wrap.

    Returns:
        function: The wrapped function.
    """
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

class always():
    """
    Class that always returns a specified value when called.
    """
    def __init__(self, val):
        """
        Initialize the always class with a value.

        Args:
            val: The value to always return.
        """
        self.val = val

    def __call__(self, *args, **kwargs):
        """
        Return the specified value.

        Returns:
            The specified value.
        """
        return self.val
    
class not_equals():
    """
    Class that checks if a value does not equal the specified value.
    """
    def __init__(self, val):
        """
        Initialize with a value.

        Args:
            val: The value to compare against.
        """
        self.val = val
    
    def __call__(self, x, *args, **kwargs):
        """
        Compare the input x with the specified value.

        Returns:
            bool: True if x is not equal to the specified value, False otherwise.
        """
        return x != self.val
    
class equals():
    """
    Class that checks if a value equals the specified value.
    """
    def __init__(self, val):
        """
        Initialize with a value.

        Args:
            val: The value to compare against.
        """
        self.val = val
    
    def __call__(self, x, *args, **kwargs):
        """
        Compare the input x with the specified value.

        Returns:
            bool: True if x is equal to the specified value, False otherwise.
        """
        return x == self.val
    
def init_zero_(layer):
    """
    Initialize the weights and bias of a torch layer to zero.

    Args:
        layer (torch.nn.Module): The layer to initialize.
    """
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

def pick_and_pop(keys, d):
    """
    Remove and return values from a dictionary based on provided keys.

    Args:
        keys (list): List of keys to remove from the dictionary.
        d (dict): The dictionary to pick from.

    Returns:
        dict: A dictionary with the specified keys and their values.
    """
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    """
    Group dictionary keys based on a condition.

    Args:
        cond (function): Condition to split dictionary.
        d (dict): The dictionary to group.

    Returns:
        tuple: Two dictionaries split based on the condition.
    """
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    """
    Check if a string begins with a specific prefix.

    Args:
        prefix (str): The prefix to check for.
        str (str): The string to check.

    Returns:
        bool: True if string starts with prefix, False otherwise.
    """
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    """
    Group dictionary items by keys that start with a specific prefix.

    Args:
        prefix (str): The prefix to check for.
        d (dict): The dictionary to group.

    Returns:
        tuple: Two dictionaries split based on the prefix condition.
    """
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    """
    Group dictionary items by keys that start with a specific prefix and remove the prefix.

    Args:
        prefix (str): The prefix to check for.
        d (dict): The dictionary to group.

    Returns:
        tuple: Dictionary with the prefix removed and another dictionary with remaining items.
    """
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


def divisible_by(num, den):
    return (num % den) == 0