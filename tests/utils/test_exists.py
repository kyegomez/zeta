import pytest
from zeta.utils import exists


def test_exists_on_none():
    assert exists(None) is False
    # Another way to write the same test
    assert not exists(None)


def test_exists_on_empty_string():
    assert exists("") is True
    assert exists(" ") is True
    # Another way to write the same test
    assert exists("")


def test_exists_on_zero():
    assert exists(0) is True
    assert exists(0.0) is True


@pytest.mark.parametrize(
    "val", [True, False, 1, -1, [], [None], {}, {"None": None}, lambda x: x]
)
def test_exists_on_values(val):
    assert exists(val) is True


def test_exists_on_function():
    assert exists(lambda x: x) is True


def test_exists_on_empty_list():
    assert exists([]) is True


def test_exists_on_empty_dict():
    assert exists({}) is True


def test_exists_on_False():
    assert exists(False) is True


def test_exists_on_None():
    assert exists(None) is False
