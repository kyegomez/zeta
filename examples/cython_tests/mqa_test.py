import timeit
import torch
from zeta import MultiQueryAttention as PyTorchMQA
from mqa import MultiQueryAttention as CythonMQA

# Input parameters
batch_size = 32
seq_len = 128
embed_dim = 512
num_heads = 8

# Create sample input tensors
query = torch.randn(batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)

# Initialize the PyTorch Multi-Query Attention layer (from zeta package)
pytorch_mqa = PyTorchMQA(dim=embed_dim, heads=num_heads)

# Initialize the Cython Multi-Query Attention layer (from mqa module)
cython_mqa = CythonMQA(embed_dim, num_heads)


# Define functions for benchmarking
def run_pytorch_mqa():
    output, _, _ = pytorch_mqa(query)
    return output


def run_cython_mqa():
    output = cython_mqa.forward(query, key, value)
    return output


# Warm-up runs (important to avoid cold start issues)
for _ in range(20):
    run_pytorch_mqa()
    run_cython_mqa()

# Benchmark PyTorch implementation
pytorch_time = timeit.timeit(
    "run_pytorch_mqa()", globals=globals(), number=1000
)

# Benchmark Cython implementation
cython_time = timeit.timeit("run_cython_mqa()", globals=globals(), number=1000)

# Print the results
print(f"PyTorch MQA execution time: {pytorch_time:.6f} seconds")
print(f"Cython MQA execution time: {cython_time:.6f} seconds")
if cython_time < pytorch_time:
    print(f"Cython is faster by: {pytorch_time / cython_time:.2f}x")
else:
    print(f"PyTorch is faster by: {cython_time / pytorch_time:.2f}x")
