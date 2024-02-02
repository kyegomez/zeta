import torch
import pytest
from zeta.models import GPT4MultiModal
from unittest.mock import patch


def test_GPT4MultiModal_initialization():
    model = GPT4MultiModal()
    assert hasattr(model, "encoder")
    assert hasattr(model, "decoder")


@pytest.fixture
def mock_model(monkeypatch):
    mock = GPT4MultiModal()
    monkeypatch.setattr("zeta.models.GPT4MultiModal", lambda: mock)
    return mock


def test_forward_successful_execution(mock_model):
    img = torch.randn(1, 3, 256, 256)
    text = torch.LongTensor([1, 2, 1, 0, 5])

    output = mock_model(img=img, text=text)
    assert output is not None


def test_forward_exception_raised(mock_model):
    with pytest.raises(Exception):
        mock_model(img=None, text=None)


@patch("zeta.models.ViTransformerWrapper")
def test_transformer_called_in_forward(mock_transformer, mock_model):
    img = torch.randn(1, 3, 256, 256)
    text = torch.LongTensor([1, 2, 1, 0, 5])
    mock_model(img, text)
    mock_transformer.assert_called_once()


@patch("zeta.models.ViTransformerWrapper", side_effect=Exception)
def test_exception_in_transformer_catch_in_forward(mock_transformer,
                                                   mock_model):
    with pytest.raises(Exception):
        mock_model(img=None, text=None)
        mock_transformer.assert_called_once()
