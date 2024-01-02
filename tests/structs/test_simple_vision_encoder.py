import torch
from zeta.structs.simple_vision_encoder import VisionEncoder


def test_simple_vision_encoder_init():
    sve = VisionEncoder()
    assert sve.size == (384, 384)
    assert sve.model_name == "vikhyatk/moondream0"
    assert sve.return_shape is False
    assert isinstance(sve.model, torch.jit.ScriptModule)
    assert sve.preprocess.transforms[-1].scale is True
    assert sve.preprocess.transforms[-1].dtype == torch.float32


def test_simple_vision_encoder_init_custom_size():
    sve = VisionEncoder(size=(512, 512))
    assert sve.size == (512, 512)


def test_simple_vision_encoder_init_custom_model_name():
    sve = VisionEncoder(model_name="custom/model")
    assert sve.model_name == "custom/model"


def test_simple_vision_encoder_init_return_shape():
    sve = VisionEncoder(return_shape=True)
    assert sve.return_shape is True
