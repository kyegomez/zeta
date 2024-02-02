from zeta.models import LLama2
from unittest.mock import Mock, patch


def test_llama2_initialization():
    mock_transformer = Mock()
    mock_autoregressive_wrapper = Mock()

    with patch("zeta.models.Transformer", return_value=mock_transformer), patch(
            "zeta.models.AutoregressiveWrapper",
            return_value=mock_autoregressive_wrapper,
    ):
        llama = LLama2()
        assert llama.llama2 == mock_transformer
        assert llama.decoder == mock_autoregressive_wrapper


def test_llama2_forward():
    mock_transformer = Mock()
    mock_autoregressive_wrapper = Mock()
    mock_forward = Mock(return_value=("model_input", "padded_x"))
    mock_autoregressive_wrapper.forward = mock_forward

    with patch("zeta.models.Transformer", return_value=mock_transformer), patch(
            "zeta.models.AutoregressiveWrapper",
            return_value=mock_autoregressive_wrapper,
    ):
        llama = LLama2()
        result = llama.forward("test text")
        mock_forward.assert_called_once_with("test text")
        mock_autoregressive_wrapper.assert_called_once_with("model_input",
                                                            padded_x="padded_x")
        assert result == mock_autoregressive_wrapper.return_value
