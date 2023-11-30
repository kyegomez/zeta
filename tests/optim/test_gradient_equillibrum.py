import pytest
import torch
from torch import nn
from torch.optim import SGD

from zeta.optim.gradient_equillibrum import GradientEquilibrum


# Helper function to create a simple model and loss for testing
def create_model_and_loss():
    dim_in = 2
    dim_out = 1
    model = torch.nn.Linear(dim_in, dim_out)
    loss_fn = torch.nn.MSELoss()
    return model, loss_fn


# Test optimizer with default parameters
def test_optimizer_default_parameters():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters())
    assert isinstance(optimizer, GradientEquilibrum)
    assert optimizer.defaults["lr"] == 0.01
    assert optimizer.defaults["max_iterations"] == 1000
    assert optimizer.defaults["tol"] == 1e-7
    assert optimizer.defaults["weight_decay"] == 0.0


# Test optimizer step function with zero gradient
def test_optimizer_step_with_zero_gradient():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters())
    optimizer.zero_grad()
    loss = loss_fn(model(torch.tensor([[0.0, 0.0]]), torch.tensor([[0.0]])))
    loss.backward()
    optimizer.step()
    assert True  # No exceptions were raised


# Test optimizer step function with a non-zero gradient
def test_optimizer_step_with_non_zero_gradient():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters())
    optimizer.zero_grad()
    loss = loss_fn(model(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])))
    loss.backward()
    optimizer.step()
    assert True  # No exceptions were raised


# Test optimizer step function with weight decay
def test_optimizer_step_with_weight_decay():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), weight_decay=0.1)
    optimizer.zero_grad()
    loss = loss_fn(model(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])))
    loss.backward()
    optimizer.step()
    assert True  # No exceptions were raised


# Test optimizer clip_grad_value function
def test_optimizer_clip_grad_value():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters())
    optimizer.zero_grad()
    loss = loss_fn(model(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])))
    loss.backward()
    optimizer.clip_grad_value(0.1)
    optimizer.step()
    assert True  # No exceptions were raised


# Test optimizer add_weight_decay function
def test_optimizer_add_weight_decay():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters())
    optimizer.add_weight_decay(0.1)
    assert optimizer.param_groups[0]["weight_decay"] == 0.1


# Test optimizer state_dict and load_state_dict functions
def test_optimizer_state_dict_and_load_state_dict():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters())
    state_dict = optimizer.state_dict()
    optimizer.load_state_dict(state_dict)
    assert optimizer.defaults == state_dict["param_groups"][0]
    assert optimizer.state == state_dict["state"]


# Test optimizer with a custom learning rate
def test_optimizer_with_custom_lr():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), lr=0.1)
    assert optimizer.defaults["lr"] == 0.1


# Test optimizer with a custom max_iterations
def test_optimizer_with_custom_max_iterations():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), max_iterations=500)
    assert optimizer.defaults["max_iterations"] == 500


# Test optimizer with a custom tolerance
def test_optimizer_with_custom_tolerance():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), tol=1e-6)
    assert optimizer.defaults["tol"] == 1e-6


# Test optimizer with a custom learning rate and weight decay
def test_optimizer_with_custom_lr_and_weight_decay():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), lr=0.1, weight_decay=0.2)
    assert optimizer.defaults["lr"] == 0.1
    assert optimizer.defaults["weight_decay"] == 0.2


# Test optimizer with a custom clip threshold
def test_optimizer_with_custom_clip_threshold():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), clip_thresh=0.5)
    assert True  # No exceptions were raised


# Test optimizer with custom parameters and custom learning rate
def test_optimizer_with_custom_parameters_and_lr():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(
        model.parameters(),
        lr=0.1,
        max_iterations=500,
        tol=1e-6,
        weight_decay=0.2,
    )
    assert optimizer.defaults["lr"] == 0.1
    assert optimizer.defaults["max_iterations"] == 500
    assert optimizer.defaults["tol"] == 1e-6
    assert optimizer.defaults["weight_decay"] == 0.2


# Test optimizer with a large learning rate and max_iterations
def test_optimizer_with_large_lr_and_max_iterations():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(
        model.parameters(), lr=1e3, max_iterations=10000
    )
    assert optimizer.defaults["lr"] == 1e3
    assert optimizer.defaults["max_iterations"] == 10000


# Test optimizer with a very small tolerance
def test_optimizer_with_small_tolerance():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), tol=1e-10)
    assert optimizer.defaults["tol"] == 1e-10


# Test optimizer step function with a custom closure
def test_optimizer_step_with_custom_closure():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters())

    # Custom closure that computes and returns loss
    def custom_closure():
        optimizer.zero_grad()
        loss = loss_fn(model(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])))
        loss.backward()
        return loss

    loss = optimizer.step(closure=custom_closure)
    assert isinstance(loss, torch.Tensor)


# Test optimizer with custom parameters and weight decay
def test_optimizer_with_custom_parameters_and_weight_decay():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(
        model.parameters(),
        lr=0.1,
        max_iterations=500,
        tol=1e-6,
        weight_decay=0.2,
    )
    assert optimizer.defaults["lr"] == 0.1
    assert optimizer.defaults["max_iterations"] == 500
    assert optimizer.defaults["tol"] == 1e-6
    assert optimizer.defaults["weight_decay"] == 0.2


# Test optimizer step function with custom learning rate
def test_optimizer_step_with_custom_lr():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss = loss_fn(model(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])))
    loss.backward()
    optimizer.step(lr=0.01)  # Custom learning rate for this step
    assert True  # No exceptions were raised


# Test optimizer step function with a very small learning rate
def test_optimizer_step_with_small_lr():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss = loss_fn(model(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])))
    loss.backward()
    optimizer.step(lr=1e-6)  # Very small learning rate for this step
    assert True  # No exceptions were raised


# Test optimizer step function with a custom clip threshold
def test_optimizer_step_with_custom_clip_threshold():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), clip_thresh=0.5)
    optimizer.zero_grad()
    loss = loss_fn(model(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])))
    loss.backward()
    optimizer.step()
    assert True  # No exceptions were raised


# Test optimizer step function with weight decay and custom learning rate
def test_optimizer_step_with_weight_decay_and_custom_lr():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), lr=0.1, weight_decay=0.2)
    optimizer.zero_grad()
    loss = loss_fn(model(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])))
    loss.backward()
    optimizer.step(lr=0.01)  # Custom learning rate for this step
    assert True  # No exceptions were raised


# Test optimizer step function with custom gradient values
def test_optimizer_step_with_custom_gradient_values():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters())
    optimizer.zero_grad()

    # Custom gradients for testing
    custom_gradients = [torch.tensor([[-1.0, -1.0]])]
    for param, grad in zip(model.parameters(), custom_gradients):
        param.grad = grad

    loss = loss_fn(model(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])))
    loss.backward()
    optimizer.step()

    # Check if the parameters were updated correctly
    for param, grad in zip(model.parameters(), custom_gradients):
        assert torch.allclose(param.data, grad, atol=1e-7)


# Test optimizer step function with custom gradient values and clip threshold
def test_optimizer_step_with_custom_gradient_values_and_clip_threshold():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), clip_thresh=0.5)
    optimizer.zero_grad()

    # Custom gradients for testing
    custom_gradients = [torch.tensor([[-1.0, -1.0]])]
    for param, grad in zip(model.parameters(), custom_gradients):
        param.grad = grad

    loss = loss_fn(model(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])))
    loss.backward()
    optimizer.step()

    # Check if the parameters were updated correctly and clipped
    for param, grad in zip(model.parameters(), custom_gradients):
        clipped_grad = torch.clamp(grad, -0.5, 0.5)
        assert torch.allclose(param.data, clipped_grad, atol=1e-7)


# Test optimizer step function with custom gradient values and weight decay
def test_optimizer_step_with_custom_gradient_values_and_weight_decay():
    model, loss_fn = create_model_and_loss()
    optimizer = GradientEquilibrum(model.parameters(), weight_decay=0.1)
    optimizer.zero_grad()

    # Custom gradients for testing
    custom_gradients = [torch.tensor([[-1.0, -1.0]])]
    for param, grad in zip(model.parameters(), custom_gradients):
        param.grad = grad

    loss = loss_fn(model(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])))
    loss.backward()
    optimizer.step()

    # Check if the parameters were updated correctly with weight decay
    for param, grad in zip(model.parameters(), custom_gradients):
        updated_param = grad - 0.1 * grad
        assert torch.allclose(param.data, updated_param, atol=1e-7)


# Define a sample model and data
class SampleModel(nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


# Define a benchmark function
@pytest.mark.benchmark(group="optimizer_comparison")
def test_optimizer_performance(benchmark):
    # Create a sample model and data
    model = SampleModel()
    data = torch.randn(64, 10)
    target = torch.randn(64, 10)
    loss_fn = nn.MSELoss()

    # Create instances of your optimizer and an alternative optimizer
    custom_optimizer = GradientEquilibrum(model.parameters(), lr=0.01)
    sgd_optimizer = SGD(model.parameters(), lr=0.01)

    # Benchmark your optimizer's step method
    def custom_step():
        custom_optimizer.zero_grad()
        loss = loss_fn(model(data), target)
        loss.backward()
        custom_optimizer.step()

    # Benchmark the alternative optimizer's step method
    def sgd_step():
        sgd_optimizer.zero_grad()
        loss = loss_fn(model(data), target)
        loss.backward()
        sgd_optimizer.step()

    # Measure and compare execution times
    custom_time = benchmark(custom_step)
    sgd_time = benchmark(sgd_step)

    # Assert that your optimizer is as fast or faster than the alternative
    assert custom_time < sgd_time
