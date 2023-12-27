import pytest
from zeta.utils import maybe


# Mock function to use for testing
def mock_func(x):
    return x * 10


def exists(item):
    return item is not None


# Test 1: Basic function call with existing argument
def test_maybe_with_existing_arg():
    @maybe
    def function_to_test(x):
        return mock_func(x)

    assert function_to_test(5) == 50


# Test 2: Function call with non-existing argument
def test_maybe_with_non_existing_arg():
    @maybe
    def function_to_test(x):
        return mock_func(x)

    assert function_to_test(None) is None


# Test 3: Function call with multiple arguments
def test_maybe_with_multiple_args():
    @maybe
    def function_to_test(x, y, z):
        return mock_func(x) + y + z

    assert function_to_test(5, 2, 3) == 55


# Test 4: Function call with keyword arguments
def test_maybe_with_keyword_args():
    @maybe
    def function_to_test(x, y=1, z=1):
        return mock_func(x) + y + z

    assert function_to_test(5, y=5, z=5) == 60


# Test 5: Parameterized testing with various inputs


@pytest.mark.parametrize("input,output", [(5, 50), (None, None), (0, 0)])
def test_maybe_parameterized(input, output):
    @maybe
    def function_to_test(x):
        return mock_func(x)

    assert function_to_test(input) == output


# Test 6: Exception testing


def test_maybe_exception_handling():
    @maybe
    def function_to_test(x):
        return x / 0

    with pytest.raises(ZeroDivisionError):
        function_to_test(5)
