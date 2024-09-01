# save_memory_snapshot

# Module Name: save_memory_snapshot

The `save_memory_snapshot` function within PyTorch is a context manager that allows developers to save memory usage snapshots from their PyTorch model to a specified file path. This is particularly useful for tracking and analyzing memory utilization during code execution, facilitating optimized resource management.  

Function Details:
```python
@contextmanager
def save_memory_snapshot(file_path: Path):
    """Save a memory snapshot information to a folder
    Usage:
        with save_memory_snapshot(file_path):
            # code to profile

    Args:
        file_path: The path to the folder to save the snapshot to
                    will create the folder if it doesn't exist
    """
    file_path.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._record_memory_history()
    try:
        yield
    finally:
        s = torch.cuda.memory._snapshot()
        with open(f"{file_path}/snapshot.pickle", "wb") as f:
            dump(s, f)
        with open(f"{file_path}/trace_plot.html", "w") as f:
            f.write(torch.cuda._memory_viz.trace_plot(s))
```
Here is a description for the single argument,  `file_path`:

| Parameter | Type | Description |
|-----------|------|-------------|
| file_path | pathlib.Path | File path to a folder where the snapshots will be saved. The function will create the folder if it does not exist. |

**Functionality and Usage**

After creating the output directory (if it does not exist), the function initiates recording the GPU's memory usage history via torch.cuda.memory._record_memory_history(). 

Any code executed within the context of the `save_memory_snapshot` function will be profiled, and memory usage snapshots during its execution will be stored. 

Upon completion of the code block within the context, a snapshot of the memory history at that point in time is captured using `torch.cuda.memory._snapshot()`. This snapshot is then saved in pickle format (`snapshot.pickle`), and a HTML file (`trace_plot.html`) is generated, displaying a trace plot for the memory usage. 

The execution flow control is then returned to the code following the context block, ensuring any code thereafter is not profiled.

**How to Use**
```python
from pathlib import Path

import torch

from zeta.utils import save_memory_snapshot

file_path = Path("my_folder")

# code to profile
model = torch.nn.Linear(10, 10)
input_tensor = torch.randn(10, 10)

with save_memory_snapshot(file_path):
    output = model(input_tensor)
```
The provided file path 'my_folder' is where the snapshots will be saved. After this code block executed, the snapshot of the memory usage by the Linear layer applied on input_tensor will be saved to 'my_folder' in both 'snapshot.pickle' file and 'trace_plot.html' file. 

**Use Case 2**
```python
from pathlib import Path

import torch

from zeta.utils import save_memory_snapshot

file_path = Path("gpu_usage")

# code to profile
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 20, 5),
    torch.nn.ReLU(),
    torch.nn.Conv2d(20, 64, 5),
    torch.nn.ReLU(),
)

input_tensor = torch.randn(1, 1, 32, 32)

with save_memory_snapshot(file_path):
    output = model(input_tensor)
```
In this case, we are profiling a multi-layer Convolutional Neural Network (CNN). The memory snapshot will give insights about the intermediate usage and fluctuations occurring due to convolutions and the subsequent ReLU activation function. 

**Use Case 3**
```python
from pathlib import Path

import torch

from zeta.utils import save_memory_snapshot

file_path = Path("training_memory")

# establish a simple model
model = torch.nn.Linear(20, 10)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# dummy data
inputs = torch.randn(10, 20)
targets = torch.randn(10, 10)

with save_memory_snapshot(file_path):
    # a complete step of training
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```
In this last example, we are profiling the memory usage during an entire step of model training, including forward pass, calculating loss, backward pass (backpropagation), and updating weights.

For each example, two files hopefully providing useful insights on memory utilization should be generated in the specified 'file_path': `snapshot.pickle` and `trace_plot.html`.
