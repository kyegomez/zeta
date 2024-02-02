import os
import pytest
import torch
import torch.nn as nn
from zeta.quant.niva import niva
from zeta.nn import QFTSPEmbedding


def test_niva_model_type():
    with pytest.raises(TypeError):
        niva(
            "not a model",
            model_path="model.pt",
            output_path="model_quantized.pt",
        )


def test_niva_model_path_none():
    model = QFTSPEmbedding(100, 100)
    with pytest.raises(ValueError):
        niva(model, model_path=None, output_path="model_quantized.pt")


def test_niva_output_path_none():
    model = QFTSPEmbedding(100, 100)
    with pytest.raises(ValueError):
        niva(model, model_path="model.pt", output_path=None)


def test_niva_quant_type_invalid():
    model = QFTSPEmbedding(100, 100)
    with pytest.raises(ValueError):
        niva(
            model,
            model_path="model.pt",
            output_path="model_quantized.pt",
            quant_type="invalid",
        )


def test_niva_quantize_layers_not_list():
    model = QFTSPEmbedding(100, 100)
    with pytest.raises(TypeError):
        niva(
            model,
            model_path="model.pt",
            output_path="model_quantized.pt",
            quantize_layers="not a list",
        )


def test_niva_quantize_layers_not_types():
    model = QFTSPEmbedding(100, 100)
    with pytest.raises(TypeError):
        niva(
            model,
            model_path="model.pt",
            output_path="model_quantized.pt",
            quantize_layers=["not a type"],
        )


def test_niva_quantize_layers_not_subclasses():
    model = QFTSPEmbedding(100, 100)
    with pytest.raises(TypeError):
        niva(
            model,
            model_path="model.pt",
            output_path="model_quantized.pt",
            quantize_layers=[str],
        )


def test_niva_dtype_not_dtype():
    model = QFTSPEmbedding(100, 100)
    with pytest.raises(TypeError):
        niva(
            model,
            model_path="model.pt",
            output_path="model_quantized.pt",
            dtype="not a dtype",
        )


def test_niva_dtype_invalid():
    model = QFTSPEmbedding(100, 100)
    with pytest.raises(ValueError):
        niva(
            model,
            model_path="model.pt",
            output_path="model_quantized.pt",
            dtype=torch.float32,
        )


def test_niva_quantize_layers_none_dynamic():
    model = QFTSPEmbedding(100, 100)
    with pytest.raises(ValueError):
        niva(
            model,
            model_path="model.pt",
            output_path="model_quantized.pt",
            quant_type="dynamic",
            quantize_layers=None,
        )


# The following tests assume that "model.pt" exists and is a valid model file
def test_niva_dynamic():
    model = QFTSPEmbedding(100, 100)
    niva(
        model,
        model_path="model.pt",
        output_path="model_quantized.pt",
        quant_type="dynamic",
        quantize_layers=[nn.Embedding],
    )


def test_niva_static():
    model = QFTSPEmbedding(100, 100)
    niva(
        model,
        model_path="model.pt",
        output_path="model_quantized.pt",
        quant_type="static",
    )


def test_niva_qint8():
    model = QFTSPEmbedding(100, 100)
    niva(
        model,
        model_path="model.pt",
        output_path="model_quantized.pt",
        dtype=torch.qint8,
    )


def test_niva_quint8():
    model = QFTSPEmbedding(100, 100)
    niva(
        model,
        model_path="model.pt",
        output_path="model_quantized.pt",
        dtype=torch.quint8,
    )


# The following tests assume that "model_quantized.pt" is the output of a previous test
def test_niva_output_exists():
    assert os.path.exists("model_quantized.pt")


def test_niva_output_loadable():
    model = QFTSPEmbedding(100, 100)
    model.load_state_dict(torch.load("model_quantized.pt"))


def test_niva_output_correct_type():
    model = QFTSPEmbedding(100, 100)
    model.load_state_dict(torch.load("model_quantized.pt"))
    assert isinstance(model, nn.Module)


def test_niva_output_quantized():
    model = QFTSPEmbedding(100, 100)
    model.load_state_dict(torch.load("model_quantized.pt"))
    assert any(
        hasattr(module, "qconfig") and module.qconfig
        for module in model.modules())
