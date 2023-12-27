import pytest
from zeta.utils import print_main
from unittest.mock import patch


# Usage of Fixtures
@pytest.fixture
def message():
    # This will create a predefined message that will be used in every test
    return "This is the test message!"


# Basic Test
def test_print_main_without_dist(message, capsys):
    """Test print_main without distribution"""
    print_main(message)
    captured = capsys.readouterr()
    assert captured.out == message + "\n"


# Utilizing Mocks and Parameterized Testing
@patch("torch.distributed.is_available")
@patch("torch.distributed.get_rank")
@pytest.mark.parametrize(
    "available,rank,expected",
    [
        (True, 0, "This is the test message!\n"),
        (True, 1, ""),
        (False, 0, "This is the test message!\n"),
    ],
)
def test_print_main_with_dist(
    mock_is_available, mock_get_rank, available, rank, expected, message, capsys
):
    mock_is_available.return_value = available
    mock_get_rank.return_value = rank
    print_main(message)
    captured = capsys.readouterr()
    assert captured.out == expected
