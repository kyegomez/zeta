import pytest

from zeta.utils import string_begins_with


# Basic Tests - 1
def test_string_begins_with_true():
    assert string_begins_with("pre", "prefix") is True


# Basic Tests - 2
def test_string_begins_with_false():
    assert string_begins_with("post", "prefix") is False


# Parameterized Testing - 3, 4
@pytest.mark.parametrize(
    "prefix, string, expected",
    [("pre", "prefix", True), ("post", "prefix", False)],
)
def test_string_begins_with_parametrized(prefix, string, expected):
    assert string_begins_with(prefix, string) == expected


# Test case sensitivity and unicode characters - 5, 6
@pytest.mark.parametrize(
    "prefix, string, expected",
    [("тест", "тестовый", True), ("Тест", "тестовый", False)],
)
def test_string_begins_with_casing(prefix, string, expected):
    assert string_begins_with(prefix, string) == expected


# Test empty strings and none inputs - 7, 8, 9, 10
@pytest.mark.parametrize(
    "prefix, string, expected",
    [
        (None, "test", False),
        ("", "test", True),
        ("test", None, False),
        ("test", "", False),
    ],
)
def test_string_begins_with_empty_none(prefix, string, expected):
    assert string_begins_with(prefix, string) == expected


# Test with numbers and special characters - 11, 12, 13, 14
@pytest.mark.parametrize(
    "prefix, string, expected",
    [
        (123, "123test", False),
        ("#$", "#$test", True),
        ("test", "@#", False),
        (None, None, False),
    ],
)
def test_string_begins_with_non_letters(prefix, string, expected):
    assert string_begins_with(prefix, string) == expected
