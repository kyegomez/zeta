# import torch
# import time
# import tracemalloc

# import matplotlib.pyplot as plt
# # from memory_profiler import profile
# from zeta import MultiheadAttentionTritonTriton as MultiheadAttentionTriton

# #set seeed 
# torch.manual_seed(0)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(0)

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


# class Args:
#     def __init__(self):
#         self.layernorm_eps = 1e-5
#         self.xpos_rel_pos = False
#         self.xpos_scale_base = 1.0
#         self.multiway = True
# args = Args()

# #initialize attention
# attention = MultiheadAttentionTriton(
#     args, 
#     embed_dim=1024,
#     num_heads=8,
#     dropout=0.0,
#     self_attention=True,
#     subln=True,
# ).to(device)

# #test the input 
# test_input = torch.randint(0, 256, (1, 1024)).unsqueeze(0).float().to(device)

# #measure forward pass time
# start_time = time.time()
# output, _ = attention(test_input, test_input, test_input)
# end_time = time.time()
# print(f"Forward pass time: {end_time - start_time} seconds:")


# #measure backward pass time
# optimizer = torch.optim.Adam(attention.parameters())
# loss_function = torch.nn.CrossEntropyLoss()
# optimizer.zero_grad()
# loss = loss_function(output, test_input)
# start_time = time.time()
# loss.backward()
# end_time = time.time()
# print(f"Backware pass time: {end_time - start_time} seconds")


# #memory usage
# tracemalloc.start()
# attention(test_input, test_input, test_input)
# current, peak = tracemalloc.get_traced_memory()
# tracemalloc.stop()
# print(f"Current memory usage: {current / 10**6}MB: Peak: {peak / 10**6}MB")


# #measure consistency
# outputs = []
# for _ in range(10):
#     outputs, _ = attention(test_input, test_input, test_input)
#     outputs.append(output.detach().cpu().numpy())
# consistency_score = sum([1 for i in range(1, 10) if (outputs[0] == outputs[i]). all()])
# print(f"Consistency score: {consistency_score}")

# #measure speed for different sequence lentgs
# sequence_lengths = [1024, 2048, 4096, 8192, 160032, 32002]
# times = []
# for length in sequence_lengths:
#     test_input = torch.randint(0, 256, (1, length)).unsqueeze(0).float().to(device)
#     start_time = time.time()
#     attention(test_input, test_input, test_input)
#     end_time = time.time()
#     times.append(end_time - start_time)

















# #########V2
# import os
# import torch
# import time
# import tracemalloc
# import matplotlib.pyplot as plt
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data import TensorDataset, DataLoader
# from zeta import MultiheadAttentionTriton

# class AttentionTester:
#     def __init__(self, attention, device):
#         self.attention = attention
#         self.device = device

#     def measure_forward_pass_time(self, input):
#         start_time = time.time()
#         output, _ = self.attention(input, input, input)
#         end_time = time.time()
#         return end_time - start_time

#     def measure_backward_pass_time(self, input, optimizer, loss_function):
#         output, _ = self.attention(input, input, input)
#         optimizer.zero_grad()
#         loss = loss_function(output, input)
#         start_time = time.time()
#         loss.backward()
#         end_time = time.time()
#         return end_time - start_time

#     def measure_memory_usage(self, input):
#         tracemalloc.start()
#         self.attention(input, input, input)
#         current, peak = tracemalloc.get_traced_memory()
#         tracemalloc.stop()
#         return current / 10**6, peak / 10**6

#     def measure_consistency(self, input):
#         outputs = []
#         for _ in range(10):
#             output, _ = self.attention(input, input, input)
#             outputs.append(output.detach().cpu().numpy())
#         consistency_score = sum([1 for i in range(1, 10) if (outputs[0] == outputs[i]).all()]) / 10 * 100
#         return consistency_score

# class DistributedTester(AttentionTester):
#     def __init__(self, attention, device, world_size):
#         super().__init__(attention, device)
#         self.world_size = world_size

#     def setup(self, rank, world_size):
#         os.environ['MASTER_ADDR'] = 'localhost'
#         os.environ['MASTER_PORT'] = '7400'
#         torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

#     def cleanup(self):
#         torch.distributed.destroy_process_group()

#     def distributed_test(self, rank, world_size, input):
#         self.setup(rank, world_size)
#         attention = self.attention.to(rank)
#         ddp_attention = DDP(attention, device_ids=[rank])
#         tensor_x = torch.Tensor(input).to(rank) 
#         tensor_y = torch.Tensor(input).to(rank)
#         my_dataset = TensorDataset(tensor_x,tensor_y)
#         sampler = DistributedSampler(my_dataset)
#         dataloader = DataLoader(my_dataset, sampler=sampler)
#         for batch_idx, (data, target) in enumerate(dataloader):
#             optimizer = torch.optim.Adam(ddp_attention.parameters())
#             loss_function = torch.nn.CrossEntropyLoss()
#             forward_time = self.measure_forward_pass_time(data)
#             backward_time = self.measure_backward_pass_time(data, optimizer, loss_function)
#             current_memory, peak_memory = self.measure_memory_usage(data)
#             consistency = self.measure_consistency(data)
#             print(f"Forward pass time: {forward_time} seconds")
#             print(f"Backward pass time: {backward_time} seconds")
#             print(f"Current memory usage: {current_memory}MB; Peak: {peak_memory}MB")
#             print(f"Consistency score: {consistency}%")
#         self.cleanup()


# class Args:
#     def __init__(self):
#         self.layernorm_eps = 1e-5
#         self.xpos_rel_pos = False
#         self.xpos_scale_base = 1.0
#         self.multiway = True

# args = Args()

# if __name__ == "__main__":
#     torch.manual_seed(0)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(0)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     attention = MultiheadAttentionTriton(
#         args,
#         embed_dim=1024,
#         num_heads=8,
#         dropout=0.0,
#         self_attention=True,
#         subln=True,
#         ).to(device)
#     tester = DistributedTester(attention, device, world_size=5)
#     test_input = torch.randint(0, 256, (1, 1024)).to(device)
#     tester.distributed_test(0, 4, test_input)








####### v3
import torch
from zeta import MultiheadAttentionTriton

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Args:
    def __init__(self):
        self.layernorm_eps = 1e-5
        self.xpos_rel_pos = False
        self.xpos_scale_base = 1.0
        self.multiway = True
args = Args()

# Define the parameters
embed_dim = 512
num_heads = 8
sequence_lengths = [2**i for i in range(10, 15)]  # sequence lengths up to 16,000+

# Initialize the MultiheadAttentionTriton
args = Args()
multihead_attention = MultiheadAttentionTriton(
    args, 
    embed_dim=embed_dim,
    num_heads=8,
    dropout=0.0,
    self_attention=True,
    subln=True,
).to(device)

for seq_len in sequence_lengths:
    # Create random input tensors
    query = torch.randn((1, seq_len, embed_dim)).unsqueeze(0).to(device)
    key = torch.randn((1, seq_len, embed_dim)).unsqueeze(0).to(device)
    value = torch.randn((1, seq_len, embed_dim)).unsqueeze(0).to(device)

    # Forward pass
    output, attn_weights = multihead_attention(query, key, value)

    # Check the output shape
    assert output.shape == (1, seq_len, embed_dim), f"Output shape mismatch for seq_len {seq_len}"
    assert attn_weights.shape == (1, num_heads, seq_len, seq_len), f"Attn_weights shape mismatch for seq_len {seq_len}"

    print(f"Passed for seq_len {seq_len}")