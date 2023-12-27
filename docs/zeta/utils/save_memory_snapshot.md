# save_memory_snapshot

# `zeta.utils`

Welcome to the documentation for `zeta.utils`, a module containing utility functions to aid in managing memory snapshots. This documentation will be divided into sections explaining what is done, the class components, its uses, parameters involved and usage examples. The latter will hold code snippets demonstrating zeta's functionalities.

## Table of Contents

- [Introduction](#Introduction)
- [Function Definition](#Function-Definition)
- [Implementation](#Implementation)
- [Example Usage](#Example-Usage)


## Introduction

Memory management becomes crucial when running computations on graphics processing units (GPUs). The `zeta.utils` module provides a context manager (`save_memory_snapshot`) to profile code execution, record the GPU memory usage and save the memory snapshot information to the specified file path.

The `save_memory_snapshot` function uses PyTorch functions for memory profiling. PyTorch functions (`torch.cuda.memory._record_memory_history()`, `torch.cuda.memory._snapshot()`) provided here are for internal use and not part of the public API; hence, you may observe variation in behavior between different PyTorch versions.

## Function Definition

The function `save_memory_snapshot` implemented in the module is defined as follows:

```python
@contextmanager
def save_memory_snapshot(file_path: Path):
```

### Parameters

| Parameters | Data Type | Description |
| ------ | ------ | ----------- |
| file_path | pathlib.Path | The path to the folder to save the snapshot to. The function will create the folder if it doesn't exist.

## Implementation

The `save_memory_snapshot()` function creates a directory at the given file path, records a history of the GPU memory usage, captures a snapshot of the memory and saves both memory history and the snapshot to a file.

Its workflow is as follows:

1. The function receives `file_path` as an input parameter.
2. It creates a new directory at `file_path` if it doesn't exist already.
3. The function records the GPU memory usage history by calling `torch.cuda.memory._record_memory_history()`.
4. Code within the function's context is executed, during which the memory usage is tracked.
5. Upon completion of the execution of this context code, a snapshot of the current GPU memory status is taken (by calling `torch.cuda.memory._snapshot()`).
6. Both memory history and snapshot are saved to files at the specified location. 

The snippet of the implementation will be like this,

```
