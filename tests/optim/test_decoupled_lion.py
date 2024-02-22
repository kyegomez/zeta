import pytest
import torch
from torch import nn

from zeta.optim.decoupled_lion import DecoupledLionW


def test_decoupledlionw_initialization():
    model = nn.Linear(10, 1)
    optimizer = DecoupledLionW(model.parameters(), lr=0.01)
    assert isinstance(optimizer, DecoupledLionW)
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_decoupledlionw_step():
    model = nn.Linear(10, 1)
    optimizer = DecoupledLionW(model.parameters(), lr=0.01)
    x = torch.randn(10)
    y = torch.randn(1)
    criterion = nn.MSELoss()
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    assert loss.item() < 1.0


@pytest.mark.parametrize("lr", [0])
def test_decoupledlionw_step_edge_cases(lr):
    model = nn.Linear(10, 1)
    optimizer = DecoupledLionW(model.parameters(), lr=lr)
    x = torch.randn(10)
    y = torch.randn(1)
    criterion = nn.MSELoss()
    with pytest.raises(Exception):
        for _ in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
