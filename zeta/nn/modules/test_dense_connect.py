import torch
import torch.nn as nn
import unittest

from zeta.nn.modules.dense_connect import DenseBlock


class DenseBlockTestCase(unittest.TestCase):
    def setUp(self):
        self.submodule = nn.Linear(10, 5)
        self.dense_block = DenseBlock(self.submodule)

    def test_forward(self):
        x = torch.randn(32, 10)
        output = self.dense_block(x)

        self.assertEqual(output.shape, (32, 15))  # Check output shape
        self.assertTrue(
            torch.allclose(output[:, :10], x)
        )  # Check if input is preserved
        self.assertTrue(
            torch.allclose(output[:, 10:], self.submodule(x))
        )  # Check submodule output

    def test_initialization(self):
        self.assertEqual(
            self.dense_block.submodule, self.submodule
        )  # Check submodule assignment

    def test_docstrings(self):
        self.assertIsNotNone(
            DenseBlock.__init__.__doc__
        )  # Check if __init__ has a docstring
        self.assertIsNotNone(
            DenseBlock.forward.__doc__
        )  # Check if forward has a docstring


if __name__ == "__main__":
    unittest.main()
