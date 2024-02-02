# test_pick_and_pop.py

import pytest
from zeta.utils import pick_and_pop


def test_simple_case():
    dictionary = {"a": 1, "b": 2, "c": 3}
    keys = ["a", "b"]
    result = pick_and_pop(keys, dictionary)
    assert result == {"a": 1, "b": 2}
    assert dictionary == {"c": 3}


def test_empty_keys():
    dictionary = {"a": 1, "b": 2, "c": 3}
    keys = []
    result = pick_and_pop(keys, dictionary)
    assert result == {}
    assert dictionary == {"a": 1, "b": 2, "c": 3}


def test_key_not_found():
    dictionary = {"a": 1, "b": 2, "c": 3}
    keys = ["a", "x"]
    with pytest.raises(KeyError):
        pick_and_pop(keys, dictionary)


@pytest.mark.parametrize(
    "dict_values,keys,expected",
    [
        ({
            "a": 1,
            "b": 2,
            "c": 3
        }, ["b", "c"], {
            "b": 2,
            "c": 3
        }),
        ({
            1: "a",
            2: "b",
            3: "c"
        }, [1, 2], {
            1: "a",
            2: "b"
        }),
        ({
            "x": "y",
            "foo": "bar"
        }, ["foo"], {
            "foo": "bar"
        }),
    ],
)
def test_various_inputs(dict_values, keys, expected):
    assert pick_and_pop(keys, dict_values) == expected


def test_duplicate_keys_in_list():
    dictionary = {"a": 1, "b": 2, "c": 3}
    keys = ["a", "b", "b"]
    with pytest.raises(KeyError):
        pick_and_pop(keys, dictionary)


def test_keys_order_in_result():
    dictionary = {"a": 1, "b": 2, "c": 3}
    keys = ["b", "a"]
    result = pick_and_pop(keys, dictionary)
    assert list(result.keys()) == keys


def test_empty_dictionary():
    dictionary = {}
    keys = ["b", "a"]
    with pytest.raises(KeyError):
        pick_and_pop(keys, dictionary)
