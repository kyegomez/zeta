import pickle
from pathlib import Path
import torch
from beartype import beartype
from beartype.typing import Optional, Callable
from torch.nn import Module


# helpers
def exists(v):
    return v is not None


@beartype
def save_load(
    save_method_name="save",
    load_method_name="load",
    config_instance_var_name="_config",
    init_and_load_classmethod_name="init_and_load",
    version: Optional[str] = None,
    pre_save_hook: Optional[Callable[[Module], None]] = None,
    post_load_hook: Optional[Callable[[Module], None]] = None,
    compress: Optional[bool] = False,
    partial_load: Optional[bool] = False,
    *args,
    **kwargs,
):
    """Base decorator for save and load methods for torch.nn.Module subclasses.

    Args:
        save_method_name (str, optional): _description_. Defaults to "save".
        load_method_name (str, optional): _description_. Defaults to "load".
        config_instance_var_name (str, optional): _description_. Defaults to "_config".
        init_and_load_classmethod_name (str, optional): _description_. Defaults to "init_and_load".
        version (Optional[str], optional): _description_. Defaults to None.
        pre_save_hook (Optional[Callable[[Module], None]], optional): _description_. Defaults to None.
        post_load_hook (Optional[Callable[[Module], None]], optional): _description_. Defaults to None.
        compress (Optional[bool], optional): _description_. Defaults to False.
        partial_load (Optional[bool], optional): _description_. Defaults to False.
    """

    def _save_load(klass):
        assert issubclass(
            klass, Module
        ), "save_load should decorate a subclass of torch.nn.Module"

        _orig_init = klass.__init__

        def __init__(self, *args, **kwargs):
            _config = pickle.dumps((args, kwargs))
            setattr(self, config_instance_var_name, _config)
            _orig_init(self, *args, **kwargs)

        def _save(self, path, overwrite=True):
            if pre_save_hook:
                pre_save_hook(self)

            path = Path(path)
            assert overwrite or not path.exists()
            pkg = dict(
                model=self.state_dict(),
                config=getattr(self, config_instance_var_name),
                version=version,
            )
            torch.save(pkg, str(path), _use_new_zipfile_serialization=compress)

        def _load(self, path, strict=True):
            path = Path(path)
            assert path.exists()
            pkg = torch.load(str(path), map_location="cpu")

            if (
                exists(version)
                and exists(pkg["version"])
                and version.parse(version) != version.parse(pkg["version"])
            ):
                self.print(f'loading saved model at version {pkg["version"]},')

            model_dict = self.state_dict()
            if partial_load:
                model_dict.update(pkg["model"])
                self.load_state_dict(model_dict, strict=strict)
            else:
                self.load_state_dict(pkg["model"], strict=strict)

            if post_load_hook:
                post_load_hook(self)

        @classmethod
        def _init_and_load_from(cls, path, strict=True):
            path = Path(path)
            assert path.exists()
            pkg = torch.load(str(path), map_location="cpu")
            assert (
                "config" in pkg
            ), "model configs were not found in this saved checkpoint"

            config = pickle.loads(pkg["config"])
            args, kwargs = config
            model = cls(*args, **kwargs)

            _load(model, path, strict=strict)
            return model

        klass.__init__ = __init__
        setattr(klass, save_method_name, _save)
        setattr(klass, load_method_name, _load)
        setattr(klass, init_and_load_classmethod_name, _init_and_load_from)

        return klass

    return _save_load
