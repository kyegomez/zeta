import pytest
import torch
from gradient_ascent import GradientAscent


def mock_module():
    """Mock a simple module to simulate parameter optimization."""
    module = torch.nn.Linear(2, 2)
    return module


@pytest.fixture
def optimizer():
    module = mock_module()
    opt = GradientAscent(module.parameters())
    return opt


def test_gradient_ascent_initialization(optimizer):
    assert optimizer.lr == 0.01
    assert optimizer.momentum == 0.9
    assert optimizer.beta == 0.999
    assert not optimizer.nesterov
    assert optimizer.clip_value is None
    assert optimizer.lr_decay is None
    assert optimizer.warmup_steps == 0
    assert optimizer.logging_interval == 10
    assert optimizer.step_count == 0
    assert isinstance(optimizer.v, dict)
    assert isinstance(optimizer.m, dict)


def test_zero_grad(optimizer):
    module = mock_module()
    output = module(torch.randn(1, 2))
    loss = output.sum()
    loss.backward()
    optimizer.zero_grad()
    for param in module.parameters():
        assert torch.equal(param.grad, torch.zeros_like(param.grad))


@pytest.mark.parametrize(
    "clip_value, grad_value, expected_grad",
    [
        (
            1.0,
            torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        ),
        (
            0.5,
            torch.tensor([[0.25, -0.25], [0.25, -0.25]]),
            torch.tensor([[0.25, -0.25], [0.25, -0.25]]),
        ),
    ],
)
def test_gradient_clipping(clip_value, grad_value, expected_grad):
    module = mock_module()
    optimizer = GradientAscent(module.parameters(), clip_value=clip_value)
    for param in module.parameters():
        param.grad = grad_value.clone()
    optimizer.step()
    for param in module.parameters():
        assert torch.equal(param.grad, expected_grad)


def test_nesterov_momentum(optimizer):
    module = mock_module()
    optimizer_nesterov = GradientAscent(module.parameters(), nesterov=True)
    assert optimizer_nesterov.nesterov


@pytest.mark.parametrize("lr, expected_lr", [(0.01, 0.01), (0.1, 0.1)])
def test_learning_rate(optimizer, lr, expected_lr):
    optimizer.lr = lr
    optimizer.step()
    assert optimizer.lr == expected_lr


def test_lr_decay(optimizer):
    optimizer.lr_decay = 0.95
    initial_lr = optimizer.lr
    optimizer.step()
    assert optimizer.lr == initial_lr * optimizer.lr_decay


def test_warmup(optimizer):
    optimizer.warmup_steps = 5
    for _ in range(5):
        optimizer.step()
    assert optimizer.step_count == 5


@pytest.mark.parametrize("step_count, logging_interval, expected_output",
                         [(10, 10, True), (5, 10, False)])
def test_logging_interval(capfd, optimizer, step_count, logging_interval,
                          expected_output):
    optimizer.logging_interval = logging_interval
    optimizer.step_count = step_count
    optimizer.step()
    captured = capfd.readouterr()
    if expected_output:
        assert f"Step: {optimizer.step_count}" in captured.out
    else:
        assert captured.out == ""
