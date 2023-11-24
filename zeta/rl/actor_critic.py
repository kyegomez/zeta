import torch
from torch import nn
import torch.nn as optim


class ActorCritic(nn.Module):
    """
    Actor Critic Network

    Parameters
    ----------
    num_inputs : int
        Number of inputs
    num_outputs : int
        Number of outputs

    Returns
    -------
    dist : torch.distributions.Categorial

    Usage:
    ------
    >>> dist, value = ActorCritic(num_inputs, num_outputs, hidden_size)(x)
    >>> action = dist.sample()
    >>> action_log_probs = dist.log_prob(action)
    >>> entropy = dist.entropy().mean()

    """

    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """Forward pass"""
        value = self.critic(x)
        probs = self.actor(x)
        dist = torch.distributions.Categorial(probs)
        return dist, value


def ppo(
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    states,
    actions,
    returns,
    advantages,
    clip_param=0.2,
):
    """
    PPO step

    Parameters
    ----------
    policy_net : ActorCritic
        Policy network
    value_net : ActorCritic
        Value network
    optimizer_policy : torch.optim
        Policy optimizer
    optimizer_value : torch.optim
        Value optimizer
    states : torch.tensor
        States
    actions : torch.tensor
        Actions
    returns : torch.tensor

    Examples:
    ---------
    >>> ppo_step(network, network, optimizer_policy, optimizer_value, states, actions, returns, advantages)


    """
    dist, _ = policy_net(states)
    old_probs = dist.log_prob(actions).detach()
    _, value = value_net(states)
    criterion = nn.MSELoss()
    loss_value = criterion(value, returns)

    optimizer_value.zero_grad()
    loss_value.backward()
    optimizer_value.step()

    for _ in range(10):
        dist, _ = policy_net(states)
        new_probs = dist.log_prob(actions)
        ratio = (new_probs - old_probs).exp()
        clip_adv = (
            torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        )
        loss_policy = -torch.min(ratio * advantages, clip_adv).mean()

        optimizer_policy.zero_grad()
        loss_policy.backward()
        optimizer_policy.step()


# import torch
# import numpy as np

# # Define the environment parameters
# num_inputs = 4
# num_outputs = 2
# hidden_size = 16

# # Create the actor-critic network
# network = ActorCritic(num_inputs, num_outputs, hidden_size)

# # Create the optimizers
# optimizer_policy = optim.Adam(network.actor.parameters())
# optimizer_value = optim.Adam(network.critic.parameters())

# # Generate some random states, actions, and returns for testing
# states = torch.randn(10, num_inputs)  # 10 states, each with `num_inputs` dimensions
# actions = torch.randint(num_outputs, (10,))  # 10 actions, each is an integer in [0, `num_outputs`)
# returns = torch.randn(10, 1)  # 10 returns, each is a scalar
# advantages = torch.randn(10, 1)  # 10 advantages, each is a scalar

# # Perform a PPO step
# ppo_step(network, network, optimizer_policy, optimizer_value, states, actions, returns, advantages)

# # The `ppo_step` function first computes the old action probabilities using the policy network.
# # These are detached from the current computation graph to prevent gradients from flowing into them during the policy update.

# # Then, it computes the value loss using the value network and the returns, and performs a value network update.

# # After that, it enters a loop where it performs multiple policy updates.
# # In each update, it computes the new action probabilities, and then the ratio of the new and old probabilities.
# # This ratio is used to compute the policy loss, which is then used to update the policy network.

# # The policy loss is computed in a way that encourages the new action probabilities to stay close to the old ones,
# # which is the key idea behind PPO's objective of taking conservative policy updates.
