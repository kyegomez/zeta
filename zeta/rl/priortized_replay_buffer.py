from sumtree import SumTree
import torch
import random


class PrioritizedReplayBuffer:
    def __init__(
        self,
        state_size,
        action_size,
        buffer_size,
        device,
        eps=1e-2,
        alpha=0.1,
        beta=0.1,
    ):
        """
        Initializes a PrioritizedReplayBuffer object.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
            buffer_size (int): The maximum capacity of the buffer.
            device (torch.device): The device to store the tensors on.
            eps (float, optional): A small constant added to the priorities to ensure non-zero probabilities. Defaults to 1e-2.
            alpha (float, optional): The exponent used to compute the priority weights. Defaults to 0.1.
            beta (float, optional): The exponent used to compute the importance sampling weights. Defaults to 0.1.
        """
        self.tree = SumTree(size=buffer_size)

        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(
            buffer_size, state_size, dtype=torch.float
        )
        self.done = torch.empty(buffer_size, dtype=torch.uint8)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

        # device
        self.device = device

    def add(self, transition):
        """
        Adds a transition to the replay buffer.

        Args:
            transition (tuple): A tuple containing the state, action, reward, next_state, and done flag.
        """
        state, action, reward, next_state, done = transition

        self.tree.add(self.max_priority, self.count)

        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the replay buffer.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            tuple: A tuple containing the batch of transitions, importance sampling weights, and tree indices.
        """
        assert (
            self.real_size >= batch_size
        ), "buffer contains fewer samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)

            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total

        weights = (self.real_size * probs) ** -self.beta

        weights = weights / weights.max()
        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device),
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        """
        Updates the priorities of the transitions in the replay buffer.

        Args:
            data_idxs (list): A list of indices corresponding to the transitions in the replay buffer.
            priorities (torch.Tensor or numpy.ndarray): The updated priorities for the corresponding transitions.
        """
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
