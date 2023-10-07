import logging
import os
from functools import partial
from math import ceil
from timeit import Timer
from typing import Callable, List, NamedTuple

import plotly.graph_objects as go
import torch

from zeta.nn.attention.dilated_attention import DilatedAttention

# Generic benchmarking parameters
BATCH_SIZE = 1
TOTAL_TOKENS = 2**25  # 32M
NUM_HEADS = 4
EMBED_DIM = 8
# Vanilla attention only
VANILLA_SEQ_LENGTHS = [2**i for i in range(13, 18)]  # 8k - 128k
# Dilated attention only
SEGMENT_LENGTHS = [8192, 16384, 32768]  # 8k - 64k
DILATED_SEQ_LENGTHS = [2**i for i in range(13, 26)]  # 8k - 32M


class BenchmarkResult(NamedTuple):
    mean: float
    std: float

    def __repr__(self):
        return f"BenchmarkResult(mean: {self.mean:.3e}, std: {self.std:.3e})"

    def __str__(self):
        return f"({self.mean:.3e} \u00B1 {self.std:.3e}) s"


def benchmark(
    fn: Callable,
    *args,
    min_total_seconds: float = 1.0,
    min_iterations: int = 2,
    **kwargs,
) -> BenchmarkResult:
    if min_iterations < 2:
        raise ValueError("min_iterations must be >= 2")

    timer = Timer(
        "fn(*args, **kwargs)",
        globals={"fn": fn, "args": args, "kwargs": kwargs},
    )
    # Run the function once to warm up
    _ = timer.repeat(number=1, repeat=1)

    times: List[float] = []
    total_time = 0.0
    num_iterations = min_iterations or 1

    while total_time < min_total_seconds:
        _times = timer.repeat(number=1, repeat=num_iterations)
        times.extend(_times)
        _total_time = sum(_times)
        total_time += _total_time

        # Estimate how many more iterations we need to run to get to 1 second
        avg_time = _total_time / num_iterations
        num_iterations = ceil((min_total_seconds - total_time) / avg_time)

    times_tensor = torch.as_tensor(times)
    return BenchmarkResult(
        mean=times_tensor.mean().item(),
        std=times_tensor.std().item(),
    )


def get_dilated_attention_for_seq_length(seq_length: int) -> DilatedAttention:
    segment_lengths: List[int] = []
    dilation_rates: List[int] = []

    for segment_length in SEGMENT_LENGTHS:
        # We can't use segment lengths larger than the sequence length.
        segment_length = min(segment_length, seq_length)
        exponent = segment_length // SEGMENT_LENGTHS[0] - 1
        dilation_rate = 2**exponent

        segment_lengths.append(segment_length)
        dilation_rates.append(dilation_rate)

    return DilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
    )


def attention_forward(x: torch.Tensor, attn: Callable):
    with torch.no_grad():
        _ = attn(x, x, x)
    torch.cuda.synchronize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Benchmark vanilla attention...")
    vanilla_results: List[BenchmarkResult] = []
    for seq_length in VANILLA_SEQ_LENGTHS:
        torch.cuda.empty_cache()
        batch_size = TOTAL_TOKENS // seq_length
        x = torch.randn(
            (batch_size, seq_length, NUM_HEADS, EMBED_DIM),
            dtype=torch.float16,
            device="cuda",
        )
        fn = partial(attention_forward)
        result = benchmark(fn, x)
        vanilla_results.append(result)
        logging.info(f"Sequence length {seq_length}: {result}")

    logging.info("Benchmark dilated attention...")
    dilated_results: List[BenchmarkResult] = []
    for seq_length in DILATED_SEQ_LENGTHS:
        torch.cuda.empty_cache()
        batch_size = TOTAL_TOKENS // seq_length
        x = torch.randn(
            (batch_size, seq_length, NUM_HEADS, EMBED_DIM),
            dtype=torch.float16,
            device="cuda",
        )
        attn = get_dilated_attention_for_seq_length(seq_length)
        fn = partial(attention_forward, attn=attn)
        result = benchmark(fn, x)
        dilated_results.append(result)
        logging.info(f"Sequence length {seq_length}: {result}")

    logging.info("Plotting results...")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=VANILLA_SEQ_LENGTHS,
            y=[r.mean for r in vanilla_results],
            error_y=dict(
                type="data",
                array=[r.std for r in vanilla_results],
                visible=True,
            ),
            name="Vanilla Attention",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=DILATED_SEQ_LENGTHS,
            y=[r.mean for r in dilated_results],
            error_y=dict(
                type="data",
                array=[r.std for r in dilated_results],
                visible=True,
            ),
            name="Dilated Attention",
        ),
    )
    fig.update_layout(
        title="Attention Benchmark (Total Tokens = 32M)",
        xaxis_title="Sequence Length",
        yaxis_title="Runtime (s)",
        xaxis_type="log",
        yaxis_type="log",
    )
    fig.write_image(os.path.join("doc", "benchmark.png"))
