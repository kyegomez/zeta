import pytest
import torch
from torch.autograd import Variable

from zeta.nn.modules import DynamicRoutingBlock

# Optional if you want to use parametrization
test_data = [
    (
        Variable(torch.randn(1, 5), requires_grad=True),
        Variable(torch.randn(1, 5), requires_grad=True),
    ),
    (
        Variable(torch.randn(10, 5), requires_grad=True),
        Variable(torch.randn(10, 5), requires_grad=True),
    ),
]


@pytest.fixture
def mock_routing_module(monkeypatch):
    # maybe you would like to mock the routing_module behavior, if it's complex or time-consuming
    def mock_forward(x):
        return torch.tensor(0.5)

    monkeypatch.setattr(
        "Reference to routing_module_class", "forward", mock_forward
    )


@pytest.mark.parametrize("input1,input2", test_data)
def test_dynamic_routing_block_forward(input1, input2, mock_routing_module):
    drb = DynamicRoutingBlock(input1, input2, mock_routing_module)

    output = drb.forward(torch.randn(1, 3))

    assert output.size() == torch.Size([1, 3])
    assert torch.allclose(output, 0.5 * input1 + 0.5 * input2)


def test_dynamic_routing_block_module_assignment():
    sb1 = torch.nn.Linear(5, 3)
    sb2 = torch.nn.Linear(5, 3)
    routing_module = torch.nn.Linear(5, 1)

    drb = DynamicRoutingBlock(sb1, sb2, routing_module)

    assert drb.sb1 is sb1
    assert drb.sb2 is sb2
    assert drb.routing_module is routing_module


# And so on... You can generate more tests based on your needs
