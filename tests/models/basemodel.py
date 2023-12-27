import pytest
import zeta.models
from zeta.models import BaseModel


def test_base_model_initialization():
    test_model = zeta.models.BaseModel()
    assert isinstance(test_model, BaseModel)


def test_base_model_forward_method():
    test_model = zeta.models.BaseModel()
    with pytest.raises(NotImplementedError):
        test_model.forward()
