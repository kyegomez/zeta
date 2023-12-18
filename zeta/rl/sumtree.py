class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def propagate(self, idx, delta_value):
      parent = (idx - 1) // 2

      while parent >= 0:
        self.nodes[parent] += delta_value
        parent = (parent - 1) // 2

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        delta_value = value - self.nodes[idx]

        self.nodes[idx] = value

        self.propagate(idx, delta_value)

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def get_priority(self, data_idx):
        tree_idx = data_idx + self.size - 1
        return self.nodes[tree_idx]
    
    
    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"
    

# # Test the sum tree 
# if __name__ == '__main__':
#     # Assuming the SumTree class definition is available

#     # Function to print the state of the tree for easier debugging
#     def print_tree(tree):
#         print("Tree Total:", tree.total)
#         print("Tree Nodes:", tree.nodes)
#         print("Tree Data:", tree.data)
#         print()

#     # Create a SumTree instance
#     tree_size = 5
#     tree = SumTree(tree_size)

#     # Add some data with initial priorities
#     print("Adding data to the tree...")
#     for i in range(tree_size):
#         data = f"Data-{i}"
#         priority = i + 1  # Priority is just a simple increasing number for this test
#         tree.add(priority, data)
#         print_tree(tree)

#     # Update priority of a data item
#     print("Updating priority...")
#     update_index = 2  # For example, update the priority of the third item
#     new_priority = 10
#     tree.update(update_index, new_priority)
#     print_tree(tree)

#     # Retrieve data based on cumulative sum
#     print("Retrieving data based on cumulative sum...")
#     cumulative_sums = [5, 15, 20]  # Test with different cumulative sums
#     for cumsum in cumulative_sums:
#         idx, node_value, data = tree.get(cumsum)
#         print(f"Cumulative Sum: {cumsum} -> Retrieved: {data} with Priority: {node_value}")
#         print()
