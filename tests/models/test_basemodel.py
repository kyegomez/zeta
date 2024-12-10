import zeta.models
from zeta.models import BaseModel


def test_base_model_initialization():
    test_model = zeta.models.BaseModel()
    assert isinstance(test_model, BaseModel)
