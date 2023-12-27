import pytest
from zeta.utils import save_load
from torch.nn import Module


class TestModule(Module):
    def __init__(self, num):
        super(TestModule, self).__init__()
        self.num = num


@pytest.fixture
def path(tmp_path):
    return tmp_path / "test_module.pkl"


class TestSaveLoad:
    def test_save_load_class_decorator(self):
        @save_load()
        class TestModuleDecorated(TestModule):
            pass

        assert hasattr(TestModuleDecorated, "save")
        assert hasattr(TestModuleDecorated, "load")
        assert hasattr(TestModuleDecorated, "init_and_load")

    def test_save_method(self, path):
        @save_load()
        class TestModuleDecorated(TestModule):
            pass

        module = TestModuleDecorated(10)
        module.save(path)
        assert path.exists()

    def test_load_method(self, path):
        @save_load()
        class TestModuleDecorated(TestModule):
            pass

        module = TestModuleDecorated(10)
        module.save(path)

        loaded_module = TestModuleDecorated(1)
        loaded_module.load(path)
        assert loaded_module.num == 10

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_save_overwrite(self, path, overwrite):
        @save_load()
        class TestModuleDecorated(TestModule):
            pass

        module = TestModuleDecorated(10)
        module.save(path)
        if not overwrite:
            with pytest.raises(AssertionError):
                module.save(path, overwrite=overwrite)

    ...
