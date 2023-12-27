import pytest
import torch
from zeta.rl.priortized_rps import (
    PrioritizedSequenceReplayBuffer,
)  


@pytest.fixture
def replay_buffer():
    state_size = 4
    action_size = 2
    buffer_size = 100
    device = torch.device("cpu")
    return PrioritizedSequenceReplayBuffer(
        state_size, action_size, buffer_size, device
    )


def test_initialization(replay_buffer):
    assert replay_buffer.eps == 1e-5
    assert replay_buffer.alpha == 0.1
    assert replay_buffer.beta == 0.1
    assert replay_buffer.max_priority == 1.0
    assert replay_buffer.decay_window == 5
    assert replay_buffer.decay_coff == 0.4
    assert replay_buffer.pre_priority == 0.7
    assert replay_buffer.count == 0
    assert replay_buffer.real_size == 0
    assert replay_buffer.size == 100
    assert replay_buffer.device == torch.device("cpu")


def test_add(replay_buffer):
    transition = (torch.rand(4), torch.rand(2), 1.0, torch.rand(4), False)
    replay_buffer.add(transition)
    assert replay_buffer.count == 1
    assert replay_buffer.real_size == 1


def test_sample(replay_buffer):
    for i in range(10):
        transition = (torch.rand(4), torch.rand(2), 1.0, torch.rand(4), False)
        replay_buffer.add(transition)

    batch, weights, tree_idxs = replay_buffer.sample(5)
    assert len(batch) == 5
    assert len(weights) == 5
    assert len(tree_idxs) == 5


def test_update_priorities(replay_buffer):
    for i in range(10):
        transition = (torch.rand(4), torch.rand(2), 1.0, torch.rand(4), False)
        replay_buffer.add(transition)

    batch, weights, tree_idxs = replay_buffer.sample(5)
    new_priorities = torch.rand(5)
    replay_buffer.update_priorities(tree_idxs, new_priorities)


def test_sample_with_invalid_batch_size(replay_buffer):
    with pytest.raises(AssertionError):
        replay_buffer.sample(101)


def test_add_with_max_size(replay_buffer):
    for i in range(100):
        transition = (torch.rand(4), torch.rand(2), 1.0, torch.rand(4), False)
        replay_buffer.add(transition)

    assert replay_buffer.count == 0
    assert replay_buffer.real_size == 100


# Additional tests for edge cases, exceptions, and more scenarios can be added as needed.
