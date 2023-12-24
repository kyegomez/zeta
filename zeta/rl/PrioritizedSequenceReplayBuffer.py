from sumtree import SumTree
import torch
import random

class PrioritizedSequenceReplayBuffer:
    def __init__(self,state_size,action_size,buffer_size,device,eps=1e-5,alpha=0.1,beta=0.1,
                 decay_window=5,
                 decay_coff=0.4,
                 pre_priority=0.7):
        self.tree = SumTree(data_size=buffer_size)
        
        # PESR params
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.
        self.decay_window = decay_window
        self.decay_coff = decay_coff
        self.pre_priority = pre_priority
        
        # buffer params
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.uint8)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

        # device
        self.device = device
        
    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)
        
    def sample(self,batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)
        
        segment = self.tree.total_priority / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)
        """
        Note:
        The priorities stored in sumtree are all times alpha
        """
        probs = priorities / self.tree.total_priority
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()
        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device)
        )
        return batch, weights, tree_idxs
    
    def update_priorities(self,data_idxs,abs_td_errors):
        """
        when we get the TD-error, we should update the transition priority p_j
        And update decay_window's transition priorities
        """
        if isinstance(abs_td_errors,torch.Tensor):
            abs_td_errors = abs_td_errors.detach().cpu().numpy()
        
        for data_idx, td_error in zip(data_idxs,abs_td_errors):
            # first update the batch: p_j
            # p_j <- max{|delta_j| + eps, pre_priority * p_j}
            old_priority = self.pre_priority * self.tree.nodes[data_idx + self.tree.size - 1]
            priority = (td_error + self.eps) ** self.alpha
            priority = max(priority,old_priority)
            self.tree.update(data_idx,priority)
            self.max_priority = max(self.max_priority,priority)
            
        # And then apply decay
        if self.count >= self.decay_window:
            # count points to the next position
            # count means the idx in the buffer and number of transition
            for i in reversed(range(self.decay_window)):
                idx = (self.count - i - 1) % self.size
                decayed_priority = priority * (self.decay_coff ** (i + 1))
                tree_idx = idx + self.tree.size - 1
                existing_priority = self.tree.nodes[tree_idx]
                self.tree.update(idx,max(decayed_priority,existing_priority))