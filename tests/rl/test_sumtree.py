import pytest
from zeta.rl.sumtree import (
    SumTree,
)  # Replace 'your_module' with the actual module where SumTree is defined


# Fixture for initializing SumTree instances with a given size
@pytest.fixture
def sum_tree():
    size = 10  # You can change the size as needed
    return SumTree(size)


# Basic tests
def test_initialization(sum_tree):
    assert sum_tree.size == 10
    assert sum_tree.count == 0
    assert sum_tree.real_size == 0
    assert sum_tree.total == 0


def test_update_and_get(sum_tree):
    sum_tree.add(5, "data1")
    assert sum_tree.total == 5
    data_idx, priority, data = sum_tree.get(5)
    assert data_idx == 0
    assert priority == 5
    assert data == "data1"


def test_add_overflow(sum_tree):
    for i in range(15):
        sum_tree.add(i, f"data{i}")
    assert sum_tree.count == 5
    assert sum_tree.real_size == 10


# Parameterized testing for various scenarios
@pytest.mark.parametrize(
    "values, expected_total",
    [
        ([1, 2, 3, 4, 5], 15),
        ([10, 20, 30, 40, 50], 150),
    ],
)
def test_multiple_updates(sum_tree, values, expected_total):
    for value in values:
        sum_tree.add(value, None)
    assert sum_tree.total == expected_total


# Exception testing
def test_get_with_invalid_cumsum(sum_tree):
    with pytest.raises(AssertionError):
        sum_tree.get(20)


# More tests for specific methods
def test_get_priority(sum_tree):
    sum_tree.add(10, "data1")
    priority = sum_tree.get_priority(0)
    assert priority == 10


def test_repr(sum_tree):
    expected_repr = f"SumTree(nodes={sum_tree.nodes}, data={sum_tree.data})"
    assert repr(sum_tree) == expected_repr


# More test cases can be added as needed
