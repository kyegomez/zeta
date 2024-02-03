import pytest
import zeta.utils


# Basic Tests
def test_return_type():
    d = {"x": 1, "y": 2, "z": 3}

    def cond(x):
        return x in ["x", "y"]

    result = zeta.utils.group_dict_by_key(cond, d)
    assert isinstance(result, tuple)


# Utilizing Fixtures
@pytest.fixture
def sample_dict():
    return {"x": 1, "y": 2, "z": 3}


def test_all_keys_grouped_right(sample_dict):
    def cond(x):
        return x in ["x", "y"]

    result = zeta.utils.group_dict_by_key(cond, sample_dict)
    assert list(result[0].keys()) == ["x", "y"]
    assert list(result[1].keys()) == ["z"]


# Parameterized Testing
@pytest.mark.parametrize(
    "cond,expected_keys",
    [
        (lambda x: x in ["x", "y"], (["x", "y"], ["z"])),
        (lambda x: x in ["x"], (["x"], ["y", "z"])),
        (lambda x: x in [], ([], ["x", "y", "z"])),
        (lambda x: x in ["x", "y", "z"], (["x", "y", "z"], [])),
    ],
)
def test_keys_parameterized(cond, expected_keys, sample_dict):
    result = zeta.utils.group_dict_by_key(cond, sample_dict)
    assert list(result[0].keys()) == expected_keys[0]
    assert list(result[1].keys()) == expected_keys[1]


# Exception Testing
def test_cond_not_callable(sample_dict):
    cond = "not callable"
    with pytest.raises(TypeError):
        zeta.utils.group_dict_by_key(cond, sample_dict)
