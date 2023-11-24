import torch
import numpy as np
from collections import deque
import random


class HindsightExperienceReplay:
    """
    Hindsight experience replay buffer.

    Parameters
    ----------
    state_dim : int
        the dimension of the state
    action_dim : int
        the dimension of the action
    buffer_size : int
        the maximum size of the buffer
    batch_size : int
        the size of the mini-batch
    goal_sampling_strategy : function
        the goal sampling strategy to use

    Example:
    import torch
    from hindsight import HindsightExperienceReplay
    from numpy import np





    # Define a goal sampling strategy
    def goal_sampling_strategy(goals):
        noise = torch.randn_like(goals) * 0.1
        return goals + noise


    # Define the dimensions of the state and action spaces, the buffer size, and the batch size
    state_dim = 10
    action_dim = 2
    buffer_size = 10000
    batch_size = 64

    # Create an instance of the HindsightExperienceReplay class
    her = HindsightExperienceReplay(
        state_dim, action_dim, buffer_size, batch_size, goal_sampling_strategy
    )

    # Store a transition
    state = np.random.rand(state_dim)
    action = np.random.rand(action_dim)
    reward = np.random.rand()
    next_state = np.random.rand(state_dim)
    done = False
    goal = np.random.rand(state_dim)
    her.store_transition(state, action, reward, next_state, done, goal)

    # Sample a mini-batch of transitions
    sampled_transitions = her.sample()
    if sampled_transitions is not None:
        states, actions, rewards, next_states, dones, goals = sampled_transitions



    """

    def __init__(
        self,
        state_dim,
        action_dim,
        buffer_size,
        batch_size,
        goal_sampling_strategy,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.goal_sampling_strategy = goal_sampling_strategy

    def store_transition(self, state, action, reward, next_state, done, goal):
        """Store and transitions"""
        transition = (state, action, reward, next_state, done, goal)
        self.buffer.append(transition)

        # Store additional transition where the goal is replaced with the achieved state
        achieved_goal = next_state
        transition = (state, action, reward, next_state, done, achieved_goal)
        self.buffer.append(transition)

    def sample(self):
        """Sample a mini-batch of transitions"""
        if len(self.buffer) < self.batch_size:
            return None

        mini_batch = random.sample(self.buffer, self.batch_size)

        states, actions, rewards, next_states, dones, goals = zip(*mini_batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1)
        goals = torch.FloatTensor(goals)

        # Apply goal sampling strategy
        goals = self.goal_sampling_strategy(goals)

        return states, actions, rewards, next_states, dones, goals

    def __len__(self):
        """Return the length of the buffer"""
        return len(self.buffer)
