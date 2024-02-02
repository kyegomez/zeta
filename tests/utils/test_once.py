# Import the necessary modules
import pytest
from unittest.mock import Mock
from zeta.utils import once


def test_once_decorator():
    """Test for once decorator."""
    mock = Mock(__name__="mock")
    mock.__module__ = "mock"
    decorated_mock = once(mock)
    assert mock.call_count == 0

    # Call the decorated function for the first time
    decorated_mock(10)
    assert mock.call_count == 1
    mock.assert_called_once_with(10)

    # Call it for the second time
    decorated_mock(20)
    assert mock.call_count == 1, "Decorated function called more than once!"

    # Call it for the third time, just to make sure
    decorated_mock(30)
    assert mock.call_count == 1, "Decorated function called more than once!"


@pytest.mark.parametrize(
    "args",
    [
        (1,),
        ("hello",),
        ([1, 2, 3],),
        ({
            "a": 1
        },),
    ],
)
def test_once_decorator_with_different_arguments(args):
    """Test once decorator with different argument types."""
    mock = Mock(__name__="mock")
    mock.__module__ = "mock"
    decorated_mock = once(mock)

    decorated_mock(*args)
    mock.assert_called_once_with(*args)


def test_once_decorator_with_exception():
    """Test once decorator where the decorated function raises an exception."""
    mock = Mock(__name__="mock", side_effect=Exception("Test Exception"))
    mock.__module__ = "mock"
    decorated_mock = once(mock)

    with pytest.raises(Exception, match="Test Exception"):
        decorated_mock(10)

    assert mock.call_count == 1

    # The function should still not be callable again even if it raised an exception the first time
    with pytest.raises(Exception, match="Test Exception"):
        decorated_mock(20)

    assert mock.call_count == 1, "Decorated function called more than once!"


def test_once_decorator_with_multiple_instances():
    """Test once decorator with multiple function instances."""
    mock1 = Mock(__name__="mock1")
    mock1.__module__ = "mock1"
    decorated_mock1 = once(mock1)

    mock2 = Mock(__name__="mock2")
    mock2.__module__ = "mock2"
    decorated_mock2 = once(mock2)

    # Call the first function
    decorated_mock1(10)
    assert mock1.call_count == 1
    assert mock2.call_count == 0

    # Call the second function
    decorated_mock2(20)
    assert mock1.call_count == 1
    assert mock2.call_count == 1

    # Call the first function again
    decorated_mock1(30)
    assert (mock1.call_count == 1
           ), "Decorated mock1 function called more than once!"

    # Call the second function again
    decorated_mock2(40)
    assert (mock2.call_count == 1
           ), "Decorated mock2 function called more than once!"
