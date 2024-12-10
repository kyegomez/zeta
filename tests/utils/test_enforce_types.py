import pytest

from zeta.utils.enforce_types import enforce_types


def test_enforce_types_with_correct_types():
    @enforce_types
    def add(a: int, b: int) -> int:
        return a + b

    assert add(1, 2) == 3


def test_enforce_types_with_incorrect_types():
    @enforce_types
    def add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(TypeError):
        add("1", "2")


def test_enforce_types_with_no_annotations():
    @enforce_types
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add("1", "2") == "12"


def test_enforce_types_with_partial_annotations():
    @enforce_types
    def add(a: int, b):
        return a + b

    assert add(1, 2) == 3

    with pytest.raises(TypeError):
        add("1", 2)
