import pytest
import torch

from zeta.optim.stable_adam import StableAdamWUnfused


# Define a simple loss function for testing
def simple_loss(params):
    return sum(torch.norm(p) for p in params)


# Test initialization and basic functionality
def test_optimizer_initialization():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(model.parameters())
    assert optimizer is not None


# Test optimizer step with a simple model and no custom scalar
def test_optimizer_step_no_custom_scalar():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(model.parameters())
    loss = simple_loss(model.parameters())
    loss.backward()
    optimizer.step()


# Test optimizer step with custom scalar
def test_optimizer_step_with_custom_scalar():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(
        model.parameters(), precision="custom_fp16", custom_scalar=65536
    )
    loss = simple_loss(model.parameters())
    (loss * 65536).backward()
    optimizer.step()


# Test optimizer step with NaN or Inf gradients
def test_optimizer_step_with_nan_or_inf_gradients():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(model.parameters())

    # Create gradients with NaN or Inf values
    for param in model.parameters():
        param.grad = torch.full_like(param, float("nan"))

    with pytest.raises(RuntimeError):
        optimizer.step()


# Test optimizer state and attributes
def test_optimizer_state_and_attributes():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(model.parameters())

    # Test optimizer state attributes
    for group in optimizer.param_groups:
        assert "step" in group
        assert group["step"] == 1
        for p in group["params"]:
            assert p in optimizer.state
            state = optimizer.state[p]
            assert "exp_avg" in state
            assert "exp_avg_sq" in state


# Test optimizer with a large number of parameters
def test_optimizer_large_parameter_set():
    model = torch.nn.Sequential(*[torch.nn.Linear(10, 10) for _ in range(100)])
    optimizer = StableAdamWUnfused(model.parameters())
    loss = simple_loss(model.parameters())
    loss.backward()
    optimizer.step()


# Test optimizer with weight decay
def test_optimizer_with_weight_decay():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(model.parameters(), weight_decay=0.2)
    loss = simple_loss(model.parameters())
    loss.backward()
    optimizer.step()


# Test optimizer with different learning rates
def test_optimizer_with_different_learning_rates():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(
        [
            {"params": model.weight, "lr": 0.001},
            {"params": model.bias, "lr": 0.01},
        ]
    )
    loss = simple_loss(model.parameters())
    loss.backward()
    optimizer.step()


# Test optimizer with different beta values
def test_optimizer_with_different_beta_values():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(model.parameters(), betas=(0.95, 0.999))
    loss = simple_loss(model.parameters())
    loss.backward()
    optimizer.step()


# Test optimizer with custom clip threshold
def test_optimizer_with_custom_clip_threshold():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(model.parameters(), clip_thresh=0.5)
    loss = simple_loss(model.parameters())
    loss.backward()
    optimizer.step()


# Test optimizer with custom epsilon
def test_optimizer_with_custom_epsilon():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(model.parameters(), eps=1e-6)
    loss = simple_loss(model.parameters())
    loss.backward()
    optimizer.step()


# Test optimizer with custom precision
def test_optimizer_with_custom_precision():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(model.parameters(), precision="custom_fp16")
    loss = simple_loss(model.parameters())
    (loss * 65536).backward()
    optimizer.step()


# Test optimizer with custom scalar and precision
def test_optimizer_with_custom_scalar_and_precision():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(
        model.parameters(), precision="custom_fp16", custom_scalar=65536
    )
    loss = simple_loss(model.parameters())
    (loss * 65536).backward()
    optimizer.step()


# Test optimizer with zero gradients
def test_optimizer_with_zero_gradients():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(model.parameters())
    optimizer.step()


# Test optimizer with a negative learning rate (should raise a ValueError)
def test_optimizer_with_negative_learning_rate():
    model = torch.nn.Linear(10, 10)
    with pytest.raises(ValueError):
        StableAdamWUnfused(model.parameters(), lr=-0.001)


# Test optimizer with a negative weight decay (should raise a ValueError)
def test_optimizer_with_negative_weight_decay():
    model = torch.nn.Linear(10, 10)
    with pytest.raises(ValueError):
        StableAdamWUnfused(model.parameters(), weight_decay=-0.1)


# Test optimizer with a negative custom scalar (should raise a ValueError)
def test_optimizer_with_negative_custom_scalar():
    model = torch.nn.Linear(10, 10)
    with pytest.raises(ValueError):
        StableAdamWUnfused(
            model.parameters(), precision="custom_fp16", custom_scalar=-65536
        )


# Test optimizer with zero gradient and custom precision (should not raise exceptions)
def test_optimizer_with_zero_gradient_and_custom_precision():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(model.parameters(), precision="custom_fp16")
    optimizer.step()


# Test optimizer with zero gradient and custom scalar and precision (should not raise exceptions)
def test_optimizer_with_zero_gradient_and_custom_scalar_and_precision():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(
        model.parameters(), precision="custom_fp16", custom_scalar=65536
    )
    optimizer.step()


# Test optimizer with large clip threshold (should not raise exceptions)
def test_optimizer_with_large_clip_threshold():
    model = torch.nn.Linear(10, 10)
    optimizer = StableAdamWUnfused(model.parameters(), clip_thresh=100.0)
    loss = simple_loss(model.parameters())
    loss.backward()
    optimizer.step()
