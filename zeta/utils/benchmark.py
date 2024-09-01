import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from pickle import dump
from typing import Callable, Optional

import torch
import torch.utils.benchmark as benchmark
from torch.cuda._memory_viz import profile_plot
from torch.profiler import ProfilerActivity, profile, record_function


@dataclass
class ProfileConfig:
    file_path: Optional[str] = None
    name: Optional[str] = None
    cuda: bool = True
    iters: int = 0
    warmup_iters: int = 0
    sync: bool = False
    extra_kwargs: dict = field(default_factory=dict)
    memory_profile_path: Optional[str] = None


def benchmark_torch_function_in_microseconds(
    func: Callable, *args, **kwargs
) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    return t0.blocked_autorange().median * 1e6


def profile_function(
    config: ProfileConfig, func: Callable, *args, **kwargs
) -> torch.profiler.profile:
    """Profile a torch function and save the result to a file"""
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)

    activities = [ProfilerActivity.CPU]
    if config.cuda:
        activities.append(ProfilerActivity.CUDA)

    if config.warmup_iters >= 0:
        for _ in range(config.warmup_iters):
            func(*args, **kwargs)
    if config.sync:
        torch.cuda.synchronize()
    name_context = (
        nullcontext() if config.name is None else record_function(config.name)
    )
    profile_memory = config.memory_profile_path is not None
    with profile(
        activities=activities,
        profile_memory=profile_memory,
        record_shapes=profile_memory,
        with_stack=profile_memory,
        **config.extra_kwargs,
    ) as prof:
        for _ in range(config.iters):
            with name_context:
                func(*args, **kwargs)
                if config.sync:
                    torch.cuda.synchronize()

    if config.file_path is not None:
        prof.export_chrome_trace(config.file_path)

    if profile_memory:
        with open(config.memory_profile_path, "w") as f:
            f.write(profile_plot(prof))

    if config.file_path is None:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    return prof


@contextmanager
def print_cuda_memory_usage():
    initial_memory = torch.cuda.memory_allocated()
    try:
        yield
    finally:
        memory_usage = torch.cuda.memory_allocated() - initial_memory
        memory_usage_gb = memory_usage / (1024**3)
        print(f"CUDA memory usage: {memory_usage_gb:.2f} GB")


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
