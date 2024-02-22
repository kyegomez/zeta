import pytest

from zeta.utils import group_by_key_prefix


def test_group_by_key_prefix():
    """
    Test that the function correctly groups dictionary
    items by keys that start with a specific prefix.
    """
    prefix = "a"
    d = {"aaa": 1, "abc": 2, "ccc": 3, "ddd": 4}

    dict1, dict2 = group_by_key_prefix(prefix, d)

    assert len(dict1) == 2, "Length of 1st dictionary matches prefix count"
    assert len(dict2) == 2, "Length of 2nd dictionary matches non-prefix count"
    assert all(
        key.startswith(prefix) for key in dict1.keys()
    ), "Prefix keys are in 1st dictionary"
    assert all(
        not key.startswith(prefix) for key in dict2.keys()
    ), "Non-prefix keys are in 2nd dictionary"


def test_group_by_key_prefix_empty_dict():
    """
    Test that the function handles empty dictionaries correctly.
    """
    result = group_by_key_prefix("a", {})
    assert result == ({}, {}), "Returns two empty dictionaries"


@pytest.mark.parametrize(
    "prefix, d, result",
    [
        ("a", {"aaa": 1, "abc": 2}, ({"aaa": 1, "abc": 2}, {})),
        ("b", {"aaa": 1, "abc": 2}, ({}, {"aaa": 1, "abc": 2})),
        ("", {"aaa": 1, "abc": 2}, ({"aaa": 1, "abc": 2}, {})),
    ],
)
def test_group_by_key_prefix_parametrized(prefix, d, result):
    """
    Test various cases using parametrized testing.
    """
    assert group_by_key_prefix(prefix, d), "Results match expected"
