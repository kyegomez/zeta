import pytest
import zeta


def test_imports():
    modules = [
        "nn",
        "structs",
        "models",
        "utils",
        "training",
        "tokenizers",
        "rl",
        "optim",
        "ops",
        "quant",
    ]
    missing_modules = []
    for module in modules:
        if not hasattr(zeta, module):
            missing_modules.append(module)

    assert (
        not missing_modules
    ), f"Modules {', '.join(missing_modules)} not found in zeta package"
