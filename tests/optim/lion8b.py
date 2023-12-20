import pytest
import torch
from zeta.optim.lion8b import DecoupledLionW_8bit


def test_optimizer_init():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    optimizer = DecoupledLionW_8bit(params)

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.99)
    assert optimizer.param_groups[0]["weight_decay"] == 0


def test_optimizer_init_invalid_lr():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    with pytest.raises(ValueError):
        DecoupledLionW_8bit(params, lr=-1)


def test_optimizer_init_invalid_betas():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    with pytest.raises(ValueError):
        DecoupledLionW_8bit(params, betas=(-1, 0.99))
    with pytest.raises(ValueError):
        DecoupledLionW_8bit(params, betas=(0.9, -1))


def test_optimizer_init_invalid_weight_decay():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    with pytest.raises(ValueError):
        DecoupledLionW_8bit(params, weight_decay=-1)


def test_step_without_closure():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    optimizer = DecoupledLionW_8bit(params)
    loss = optimizer.step()

    assert loss is None


def test_step_with_closure():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    optimizer = DecoupledLionW_8bit(params)
    closure = lambda: torch.sum(params[0] ** 2 + params[1] ** 2)
    loss = optimizer.step(closure)

    assert loss is not None
    assert loss == closure()


def test_step_param_no_grad():
    params = [torch.randn(3, 3, requires_grad=False) for _ in range(2)]
    optimizer = DecoupledLionW_8bit(params)
    optimizer.step_param(params[0], optimizer.param_groups[0])

    assert params[0].grad is None


def test_step_param_with_grad():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    optimizer = DecoupledLionW_8bit(params)
    closure = lambda: torch.sum(params[0] ** 2 + params[1] ** 2)
    closure().backward()
    optimizer.step_param(params[0], optimizer.param_groups[0])

    assert params[0].grad is not None


def test_step_param_not_cuda():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    optimizer = DecoupledLionW_8bit(params, quantize=True)
    closure = lambda: torch.sum(params[0] ** 2 + params[1] ** 2)
    closure().backward()

    with pytest.raises(NotImplementedError):
        optimizer.step_param(params[0], optimizer.param_groups[0])


def test_optimizer_init_invalid_weight_decay():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    with pytest.raises(ValueError):
        DecoupledLionW_8bit(params, weight_decay=-1)


def test_step_without_closure():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    optimizer = DecoupledLionW_8bit(params)
    loss = optimizer.step()

    assert loss is None


def test_step_with_closure():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    optimizer = DecoupledLionW_8bit(params)
    closure = lambda: torch.sum(params[0] ** 2 + params[1] ** 2)
    loss = optimizer.step(closure)

    assert loss is not None
    assert loss == closure()


def test_step_param_no_grad():
    params = [torch.randn(3, 3, requires_grad=False) for _ in range(2)]
    optimizer = DecoupledLionW_8bit(params)
    optimizer.step_param(params[0], optimizer.param_groups[0])

    assert params[0].grad is None


def test_step_param_with_grad():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    optimizer = DecoupledLionW_8bit(params)
    closure = lambda: torch.sum(params[0] ** 2 + params[1] ** 2)
    closure().backward()
    optimizer.step_param(params[0], optimizer.param_groups[0])

    assert params[0].grad is not None


def test_step_param_not_cuda():
    params = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    optimizer = DecoupledLionW_8bit(params, quantize=True)
    closure = lambda: torch.sum(params[0] ** 2 + params[1] ** 2)
    closure().backward()

    with pytest.raises(NotImplementedError):
        optimizer.step_param(params[0], optimizer.param_groups[0])
