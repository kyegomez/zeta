import pytest
from zeta.utils import cast_tuple


# Basic Tests
def test_cast_tuple():
    assert cast_tuple(5, 3) == (5, 5, 5)
    assert cast_tuple("a", 2) == ("a", "a")
    assert cast_tuple((1, 2), 1) == (1, 2)


# Utilize Fixture
@pytest.fixture
def sample_value():
    return 10


def test_cast_tuple_with_fixture(sample_value):
    assert cast_tuple(sample_value, 4) == (10, 10, 10, 10)


# Parameterized Testing
@pytest.mark.parametrize(
    "value, depth, expected", [(7, 3, (7, 7, 7)), ("b", 2, ("b", "b"))]
)
def test_cast_tuple_parametrized(value, depth, expected):
    assert cast_tuple(value, depth) == expected


# Exception Testing
def test_cast_tuple_exception():
    with pytest.raises(TypeError):
        cast_tuple(5, "a")


# Test with mock and monkeypatch
def test_cast_tuple_with_mock_and_monkeypatch(monkeypatch):
    def mock_isinstance(val, t):
        return False

    monkeypatch.setattr("builtins.isinstance", mock_isinstance)
    assert cast_tuple((1, 2), 1) == ((1, 2),)
