"""Test cases for the main module of the cloud package."""

from unittest.mock import MagicMock, patch

import pytest

from zeta.cloud.main import zetacloud


@patch("zeta.cloud.main.skyapi")
@patch("zeta.cloud.main.logger")
def test_zetacloud_basic(mock_logger, mock_skyapi):
    # Arrange
    mock_task = MagicMock()
    mock_skyapi.create_task.return_value = mock_task

    # Act
    zetacloud(task_name="test_task")

    # Assert
    mock_skyapi.create_task.assert_called_once_with(
        name="test_task",
        setup="pip install requirements.txt",
        run="python train.py",
        workdir=".",
    )
    mock_logger.info.assert_called_with(f"Task: {mock_task} has been created")
    mock_task.set_resources.assert_called_once()
    mock_skyapi.launch.assert_called_once_with(mock_task, "[ZetaTrainingRun]")


# ... replicate this test with different arguments for thoroughness


@patch("zeta.cloud.main.skyapi")
@patch("zeta.cloud.main.logger")
def test_zetacloud_with_stop(mock_logger, mock_skyapi):
    # Arrange
    mock_task = MagicMock()
    mock_skyapi.create_task.return_value = mock_task

    # Act
    zetacloud(task_name="test_task", stop=True)

    # Assert
    mock_skyapi.stop.assert_called_once_with("[ZetaTrainingRun]")
    mock_logger.info.assert_called_with(
        "Cluster: [ZetaTrainingRun] has been stopped"
    )


@patch("zeta.cloud.main.skyapi")
@patch("zeta.cloud.main.logger")
def test_zetacloud_with_down(mock_logger, mock_skyapi):
    # Arrange
    mock_task = MagicMock()
    mock_skyapi.create_task.return_value = mock_task

    # Act
    zetacloud(task_name="test_task", down=True)

    # Assert
    mock_skyapi.down.assert_called_once_with("[ZetaTrainingRun]")
    mock_logger.info.assert_called_with(
        "Cluster: [ZetaTrainingRun] has been deleted"
    )


@patch("zeta.cloud.main.skyapi")
@patch("zeta.cloud.main.logger")
def test_zetacloud_with_status_report(mock_logger, mock_skyapi):
    # Arrange
    mock_task = MagicMock()
    mock_skyapi.create_task.return_value = mock_task

    # Act
    zetacloud(task_name="test_task", status_report=True)

    # Assert
    mock_skyapi.status.assert_called_once_with(
        cluster_names=["[ZetaTrainingRun]"]
    )
    mock_logger.info.assert_called_with(
        "Cluster: [ZetaTrainingRun] has been reported on"
    )


@patch("zeta.cloud.main.skyapi")
@patch("zeta.cloud.main.logger")
def test_zetacloud_with_exception(mock_logger, mock_skyapi):
    # Arrange
    mock_skyapi.create_task.side_effect = Exception("Test exception")

    # Act
    with pytest.raises(Exception):
        zetacloud(task_name="test_task")

    # Assert
    mock_logger.error.assert_called_once()


# ... replicate similar tests with minor changes for thoroughness
# Examples: test different cloud providers, test other parameter combinations, etc.
