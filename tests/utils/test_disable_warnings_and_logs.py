import logging
import os
import warnings
from unittest.mock import MagicMock, patch

from zeta.utils import disable_warnings_and_logs


@patch("logging.getLogger")
def test_warnings_disabled(mock_getLogger):
    disable_warnings_and_logs()
    warnings.filterwarnings.assert_called_once_with("ignore")
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "2"


@patch("warnings.filterwarnings")
def test_tf_warnings_disabled(mock_filterwarnings):
    disable_warnings_and_logs()
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "2"


@patch("os.environ")
def test_bnb_and_others_disabled(mock_environ):
    with patch.object(
        logging, "getLogger", return_value=MagicMock()
    ) as mock_getLogger:
        disable_warnings_and_logs()
    mock_environ.__setitem__.assert_called_once_with(
        "TF_CPP_MIN_LOG_LEVEL", "2"
    )
    mock_getLogger().setLevel.assert_called_once_with(logging.WARNING)


@patch("zeta.utils.logging")
def test_specific_loggers_disabled(mock_logging):
    mock_logger = MagicMock()
    mock_logging.getLogger.return_value = mock_logger
    disable_warnings_and_logs()
    mock_logging.getLogger.assert_any_call("real_accelerator")
    mock_logging.getLogger.assert_any_call(
        "torch.distributed.elastic.multiprocessing.redirects"
    )
    assert mock_logger.setLevel.call_count == 2
    mock_logger.setLevel.assert_called_with(logging.CRITICAL)


# @patch('logging.getLogger')
# def test_all_loggers_disabled(mock_getLogger):
#     mock_logger = MagicMock()
#     mock_getLogger.return_value = mock_logger
#     disable_warnings_and_logs()
#     mock_getLogger.assert_called()
#     mock_logger.addFilter.assert_called()
#     assert isinstance(mock_logger.addFilter.call_args[0][0], disable_warnings_and_logs.__globals__['CustomFilter'])
#     mock_getLogger().setLevel.assert_called_once_with(logging.WARNING)
#     mock_logging.disable.assert_called_once_with(logging.CRITICAL)
