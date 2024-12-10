import pytest
import torch
from torch.nn import Module

from zeta.utils.save_load_wrapper import save_load


@save_load()
class DummyModule(Module):
    def __init__(self, x):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor(x))


def test_save_load_init():
    module = DummyModule(5)
    assert isinstance(module, DummyModule)


def test_save_load_save(tmp_path):
    module = DummyModule(5)
    module.save(tmp_path / "model.pth")
    assert (tmp_path / "model.pth").exists()


def test_save_load_load(tmp_path):
    module = DummyModule(5)
    module.save(tmp_path / "model.pth")
    loaded_module = DummyModule(0)
    loaded_module.load(tmp_path / "model.pth")
    assert loaded_module.x.item() == 5


def test_save_load_init_and_load(tmp_path):
    module = DummyModule(5)
    module.save(tmp_path / "model.pth")
    loaded_module = DummyModule.init_and_load(tmp_path / "model.pth")
    assert loaded_module.x.item() == 5


def test_save_load_save_overwrite(tmp_path):
    module = DummyModule(5)
    module.save(tmp_path / "model.pth")
    with pytest.raises(AssertionError):
        module.save(tmp_path / "model.pth", overwrite=False)


def test_save_load_load_nonexistent(tmp_path):
    module = DummyModule(5)
    with pytest.raises(AssertionError):
        module.load(tmp_path / "model.pth")


def test_save_load_init_and_load_nonexistent(tmp_path):
    with pytest.raises(AssertionError):
        DummyModule.init_and_load(tmp_path / "model.pth")


def test_save_load_partial_load(tmp_path):
    @save_load(partial_load=True)
    class PartialModule(Module):
        def __init__(self, x, y):
            super().__init__()
            self.x = torch.nn.Parameter(torch.tensor(x))
            self.y = torch.nn.Parameter(torch.tensor(y))

    module = PartialModule(5, 10)
    module.save(tmp_path / "model.pth")
    loaded_module = PartialModule(0, 0)
    loaded_module.load(tmp_path / "model.pth")
    assert loaded_module.x.item() == 5
    assert loaded_module.y.item() == 0
