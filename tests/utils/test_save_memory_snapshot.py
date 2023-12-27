from unittest.mock import patch, MagicMock
from pathlib import Path
from zeta.utils import save_memory_snapshot


def test_snapshot_folder_creation():
    """Mock the Path.mkdir method to test if the folder is created"""
    with patch.object(Path, "mkdir") as mock_mkdir:
        with save_memory_snapshot(Path("/tmp")):
            pass
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_snapshot_record_start():
    """Mock the torch.cuda.memory._record_memory_history method to test if the memory history recording starts"""
    with patch("torch.cuda.memory._record_memory_history") as mock_record:
        with save_memory_snapshot(Path("/tmp")):
            pass
        mock_record.assert_called_once()


@patch("builtins.open", new_callable=MagicMock)
@patch("torch.cuda.memory._snapshot")
def test_snapshot_representation_saved(mock_snapshot, mock_open):
    """Test if the memory snapshot representation is correctly saved"""
    snapshot = {"foo": "bar"}
    mock_snapshot.return_value = snapshot

    with save_memory_snapshot(Path("/tmp")):
        pass

    mock_open.assert_called_with("/tmp/snapshot.pickle", "wb")
    f = mock_open.return_value.__enter__.return_value
    f.write.assert_called_once_with(snapshot)


@patch("builtins.open", new_callable=MagicMock)
@patch("torch.cuda.memory._snapshot")
@patch("torch.cuda._memory_viz.trace_plot")
def test_trace_plot_saved(mock_trace_plot, mock_snapshot, mock_open):
    """Test if the memory usage trace plot is correctly saved"""
    snapshot = {"foo": "bar"}
    trace_plot = "<html></html>"
    mock_snapshot.return_value = snapshot
    mock_trace_plot.return_value = trace_plot

    with save_memory_snapshot(Path("/tmp")):
        pass

    mock_open.assert_called_with("/tmp/trace_plot.html", "w")
    f = mock_open.return_value.__enter__.return_value
    f.write.assert_called_once_with(trace_plot)
