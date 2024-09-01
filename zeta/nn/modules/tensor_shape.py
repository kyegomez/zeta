import torch
from torch import Tensor


# Define the TensorShape class
class TensorShape(Tensor):
    """
    Represents the shape of a tensor.

    Args:
        data (array-like): The data of the tensor.
        shape_string (str): The string representation of the shape.

    Attributes:
        shape_string (str): The string representation of the shape.
        shape_dict (dict): A dictionary mapping dimensions to sizes.

    Raises:
        ValueError: If the shape string does not match the actual shape.

    Example:
        >>> data = [1, 2, 3, 4]
        >>> shape_string = "2 2"
        >>> tensor_shape = TensorShape(data, shape_string)
        >>> print(tensor_shape)
        TensorShape(shape_string='2 2', actual_shape=(2, 2))
    """

    def __new__(cls, data, shape_string):
        instance = torch.as_tensor(data).as_subclass(cls)
        instance.shape_string = shape_string
        instance.shape_dict = cls.parse_shape_string(
            shape_string, instance.shape
        )
        return instance

    @staticmethod
    def parse_shape_string(shape_string, actual_shape):
        """
        Parses the shape string and returns a dictionary mapping dimensions to sizes.

        Args:
            shape_string (str): The string representation of the shape.
            actual_shape (tuple): The actual shape of the tensor.

        Returns:
            dict: A dictionary mapping dimensions to sizes.

        Raises:
            ValueError: If the number of dimensions in the shape string does not match the actual shape.
        """
        dimensions = shape_string.split()
        if len(dimensions) != len(actual_shape):
            raise ValueError(
                f"Shape string {shape_string} does not match actual shape {actual_shape}"
            )
        return {dim: size for dim, size in zip(dimensions, actual_shape)}

    def __repr__(self):
        return f"TensorShape(shape_string={self.shape_string}, actual_shape={super().shape})"

    @staticmethod
    def check_shape(tensor, shape_string):
        """
        Checks if the shape of the given tensor matches the specified shape string.

        Args:
            tensor (Tensor): The tensor to check the shape of.
            shape_string (str): The string representation of the expected shape.

        Raises:
            ValueError: If the shape of the tensor does not match the expected shape.
        """
        shape_dict = TensorShape.parse_shape_string(shape_string, tensor.shape)
        if tensor.shape != tuple(shape_dict.values()):
            raise ValueError(
                f"Expected shape {shape_dict}, but got {tensor.shape}"
            )


# Define a decorator for shape checking
def check_tensor_shape(shape_string: str = None):
    """
    Decorator function that checks if the shape of a tensor matches the specified shape string.

    Args:
        shape_string (str): A string representing the desired shape of the tensor.

    Returns:
        function: A decorator function that wraps the original function and performs the shape check.

    Example:
        @check_tensor_shape("B S D")
        def my_function(tensor):
            # Function implementation
            pass

        The above example will ensure that the tensor passed to `my_function` has a shape of (2, 3).
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Assuming the tensor is the first argument
            tensor = args[1]
            TensorShape.check_shape(tensor, shape_string)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Define a helper function to create TensorShape objects
def create_tensor(
    data: Tensor = None, shape_string: str = None, random_on: bool = False
):
    if random_on:
        data = torch.randn(data)
        return TensorShape(data, shape_string)
    else:
        return TensorShape(data, shape_string)
