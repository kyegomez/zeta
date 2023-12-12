import numpy as np
import subprocess
import torch

try:
    import jax.numpy as jnp
except ImportError:
    print("JAX not installed")
    print("Installing JAX")
    subprocess.run(["pip3", "install", "jax"])
    subprocess.run(["pip3", "install", "jaxlib"])

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed")
    print("Installing Tensorflow")
    subprocess.run(["pip3", "install", "tensorflow"])


class Matrix:
    """Matrix class that can be converted between frameworks


    Args:
        data (torch.Tensor, jnp.ndarray, tf.Tensor): Data to be converted

    Example:
    >>> import torch
    >>> import jax.numpy as jnp
    >>> import tensorflow as tf
    >>> from zeta.nn.modules.matrix import Matrix
    >>>
    >>> tensor1 = Matrix(torch.tensor([1, 2, 3]))
    >>> tensor2 = Matrix(jnp.array([1, 2, 3]))
    >>> tensor3 = Matrix(tf.constant([1, 2, 3]))
    >>>
    >>> print(tensor1.to_jax())
    >>> print(tensor2.to_pytorch())
    >>> print(tensor3.to_tensorflow())


    """

    def __init__(self, data):
        self.data = data
        self.framework = self._detect_framework(data)

    def _detect_framework(self, data):
        """Detect framework

        Args:
            data (_type_): _description_

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        if isinstance(data, torch.Tensor):
            return "pytorch"
        elif isinstance(data, jnp.ndarray):
            return "jax"
        elif isinstance(data, tf.Tensor):
            return "tensorflow"
        else:
            raise TypeError("Unknown framework")

    def to_pytorch(self):
        """TODO: Docstring for to_pytorch.

        Returns:
            _type_: _description_
        """
        if self.framework == "pytorch":
            return self.data
        elif self.framework == "jax":
            # Convert JAX array to numpy array first, then to PyTorch tensor
            numpy_data = np.array(self.data)  # Convert JAX array to numpy array
            return torch.tensor(
                numpy_data
            )  # Convert numpy array to PyTorch tensor
        elif self.framework == "tensorflow":
            return torch.tensor(self.data.numpy())

    def to_jax(self):
        """To jax

        Returns:
            _type_: _description_
        """
        if self.framework == "jax":
            return self.data
        elif self.framework == "pytorch":
            return jnp.array(self.data.cpu().numpy())
        elif self.framework == "tensorflow":
            return jnp.array(self.data.numpy())

    def to_tensorflow(self):
        """To tensorflow

        Returns:
            _type_: _description_
        """
        if self.framework == "tensorflow":
            return self.data
        elif self.framework == "pytorch":
            return tf.convert_to_tensor(self.data.numpy.cpu().numpy())
        elif self.framework == "jax":
            return tf.convert_to_tensor(self.data)

    def sum(self):
        """Sum

        Returns:
            _type_: _description_
        """
        if self.framework == "pytorch":
            return self.data.sum()
        elif self.framework == "jax":
            return jnp.sum(self.data)
        elif self.framework == "tensorflow":
            return tf.reduce_sum(self.data)


# # Example usage
# tensor1 = Matrix(torch.tensor([1, 2, 3]))
# tensor2 = Matrix(jnp.array([1, 2, 3]))
# tensor3 = Matrix(tf.constant([1, 2, 3]))

# print(tensor1.to_jax())
# print(tensor2.to_pytorch())
# print(tensor3.to_tensorflow())
