import pytest
from zeta.utils import default


# Basic test
def test_default():
    assert default(None, "default") == "default"
    assert default("value", "default") == "value"


# Utilize Fixtures
@pytest.fixture
def default_params():
    return [
        ("value", "default", "value"),
        (None, "default", "default"),
        (0, "default", 0),
        (False, "default", False),
    ]


def test_default_with_params(default_params):
    for val, d, expected in default_params:
        assert default(val, d) == expected


# Parameterized Testing
@pytest.mark.parametrize(
    "val, d, expected",
    [
        ("value", "default", "value"),
        (None, "default", "default"),
        (0, "default", 0),
        (False, "default", False),
    ],
)
def test_default_parametrized(val, d, expected):
    assert default(val, d) == expected


# Exception testing
def test_default_exception():
    with pytest.raises(TypeError):
        default()


# Grouping and Marking Tests
@pytest.mark.value
def test_default_value():
    assert default("value", "default") == "value"


@pytest.mark.none
def test_default_none():
    assert default(None, "default") == "default"


# Clean Code Practices & Documentation
def test_default_value():
    """
    Test that the default function returns the correct value when one is provided.
    """
    assert default("value", "default") == "value"


def test_default_none():
    """
    Test that the default function correctly handles None values.
    """
    assert default(None, "default") == "default"


# Continue adding more tests to cover all edge cases and normal uses...
