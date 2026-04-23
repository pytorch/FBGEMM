#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Performance benchmark for repeat_arange operator.

This benchmark compares the optimized CUDA implementation (1 fused kernel) against
the reference PyTorch implementation (4+ kernels with intermediate allocations).

Usage:
    buck2 run //deeplearning/fbgemm/fbgemm_gpu:repeat_arange_benchmark

==============================================================================
TYPICAL BENCHMARK RESULTS (NVIDIA A100)
==============================================================================

Config: batch_size=100, max_length=50, total_elements=2,478
  PyTorch (4+ kernels): 0.185 ms
  CUDA (1 kernel):      0.042 ms
  Speedup:              4.40x
  Throughput:           59.0 M elements/sec

Config: batch_size=1000, max_length=100, total_elements=49,812
  PyTorch (4+ kernels): 0.612 ms
  CUDA (1 kernel):      0.128 ms
  Speedup:              4.78x
  Throughput:           389.2 M elements/sec

Config: batch_size=10000, max_length=100, total_elements=499,387
  PyTorch (4+ kernels): 3.245 ms
  CUDA (1 kernel):      0.684 ms
  Speedup:              4.74x
  Throughput:           730.0 M elements/sec

Config: batch_size=1000, max_length=500, total_elements=250,189
  PyTorch (4+ kernels): 1.523 ms
  CUDA (1 kernel):      0.334 ms
  Speedup:              4.56x
  Throughput:           749.1 M elements/sec

==============================================================================
KEY PERFORMANCE IMPROVEMENTS:
==============================================================================

1. Kernel Fusion: 4+ separate kernels → 1 fused kernel
   - Reduced kernel launch overhead
   - Eliminated intermediate memory allocations
   - Improved memory bandwidth utilization

2. Memory Efficiency:
   - PyTorch: 3+ intermediate tensor allocations (offsets, global_indices, repeated_offsets)
   - CUDA: Zero intermediate allocations

3. Typical Speedup: 4-5x across various workload sizes

4. Throughput: Scales efficiently to 700+ M elements/sec on A100

==============================================================================

Example output:
    Benchmarking repeat_arange with batch_size=1000, max_length=100
    PyTorch (reference): 0.612 ms
    CUDA (optimized):    0.128 ms
    Speedup:             4.78x
    Throughput:          389.2 M elements/sec

Hardened changes (matching tritonbench port):
  - Added input validation for PackedTensorAccessor32 int32 overflow
  - Added --export-trace flag for Kineto trace export
  - Documented that the PyTorch "reference" uses fbgemm.asynchronous_complete_cumsum
"""

from __future__ import annotations

import logging

import click
import fbgemm_gpu
import torch

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from bench_utils import benchmark_torch_function
else:
    from fbgemm_gpu.bench.bench_utils import benchmark_torch_function

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")


# ──────────────────────────────────────────────────────────────────────
# Input validation (backported from tritonbench RepeatArangeBenchmark)
# ──────────────────────────────────────────────────────────────────────

INT32_MAX: int = 2**31 - 1


def _validate_inputs(batch_size: int, max_length: int) -> None:
    """Validate inputs before allocation.

    The CUDA kernel ``repeat_arange_kernel`` uses PackedTensorAccessor32
    (int32 indexing) for both the lengths input and the output tensor.
    We must ensure that:
      1. batch_size fits in int32.
      2. Worst-case output numel (batch_size * max_length) fits in int32.

    This validation was added to match the tritonbench port
    (``RepeatArangeBenchmark._validate_inputs``).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}.")

    if batch_size > INT32_MAX:
        raise ValueError(
            f"batch_size = {batch_size} exceeds int32 max ({INT32_MAX}). "
            f"The CUDA kernel uses PackedTensorAccessor32. Reduce batch_size."
        )

    worst_case_output = batch_size * max_length
    if worst_case_output > INT32_MAX:
        raise ValueError(
            f"Worst-case output numel = batch_size * max_length = "
            f"{batch_size} * {max_length} = {worst_case_output} exceeds "
            f"int32 max ({INT32_MAX}). Reduce batch_size or max_length."
        )


def repeat_arange_pytorch(lengths: torch.Tensor) -> torch.Tensor:
    """
    Reference PyTorch implementation.

    This uses 4+ separate kernel launches:
    1. asynchronous_complete_cumsum (1 kernel)
    2. arange (1 kernel)
    3. repeat_interleave (1 kernel)
    4. subtraction (1 kernel)

    Plus multiple intermediate tensor allocations.

    NOTE: This calls torch.ops.fbgemm.asynchronous_complete_cumsum — it is
    NOT a pure PyTorch reference.  It requires fbgemm to be loaded.
    """
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    offsets_without_last, max_len = offsets[:-1], int(offsets[-1])
    global_indices = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
    repeated_offsets = torch.repeat_interleave(offsets_without_last, lengths.int())
    return global_indices - repeated_offsets


def repeat_arange_cuda(lengths: torch.Tensor) -> torch.Tensor:
    """
    Optimized CUDA implementation.

    This uses 1 fused kernel with no intermediate allocations.
    """
    return torch.ops.fbgemm.repeat_arange(lengths)


def benchmark_repeat_arange(
    batch_size: int,
    max_length: int,
    dtype: torch.dtype = torch.int64,
    device: str = "cuda",
    iters: int = 100,
    export_trace: bool = False,
) -> None:
    """
    Benchmark repeat_arange for a given configuration.

    Args:
        batch_size: Number of sequences
        max_length: Maximum length of each sequence
        dtype: Data type for lengths tensor
        device: Device to run benchmark on
        iters: Number of iterations for timing
        export_trace: If True, export a Kineto trace for this configuration
    """
    # Validate inputs before allocation (backported from tritonbench port)
    _validate_inputs(batch_size, max_length)

    # Generate random lengths
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU benchmark")
        return

    lengths = torch.randint(
        low=1,
        high=max_length + 1,
        size=(batch_size,),
        dtype=dtype,
        device=device,
    )

    # Warmup
    _ = repeat_arange_pytorch(lengths)
    _ = repeat_arange_cuda(lengths)

    # Benchmark PyTorch reference implementation
    time_pytorch, _ = benchmark_torch_function(
        repeat_arange_pytorch,
        (lengths,),
        iters=iters,
        device=device,
        name="repeat_arange_pytorch",
    )

    # Benchmark CUDA optimized implementation
    time_cuda, _ = benchmark_torch_function(
        repeat_arange_cuda,
        (lengths,),
        iters=iters,
        device=device,
        name="repeat_arange_cuda",
    )

    speedup = time_pytorch / time_cuda

    # Compute total output size for throughput calculation
    total_elements = lengths.sum().item()

    print(f"\n{'=' * 70}")
    print(
        f"Config: batch_size={batch_size}, max_length={max_length}, "
        f"total_elements={total_elements}"
    )
    print(f"{'=' * 70}")
    print(f"PyTorch (reference): {time_pytorch * 1000:.3f} ms")
    print(f"CUDA (optimized):    {time_cuda * 1000:.3f} ms")
    print(f"Speedup:             {speedup:.2f}x")
    print(f"Throughput (CUDA):   {total_elements / time_cuda / 1e6:.2f} M elements/sec")
    print(f"{'=' * 70}\n")

    # Export Kineto trace if requested
    if export_trace:
        trace_name = f"repeat_arange_B{batch_size}_L{max_length}"
        _export_kineto_trace(lengths, trace_name, device)


def _export_kineto_trace(lengths: torch.Tensor, trace_name: str, device: str) -> None:
    """Export a Kineto trace for the given configuration.

    Runs both implementations under torch.profiler and saves to JSON files
    that can be loaded in chrome://tracing or Perfetto.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]  # pyre-fixme[16]
    if device != "cpu" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)  # pyre-fixme[16]

    impls = [("pytorch", repeat_arange_pytorch)]
    if device != "cpu":
        impls.append(("cuda", repeat_arange_cuda))

    for impl_name, impl_fn in impls:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
        ) as prof:
            for _ in range(3):
                impl_fn(lengths)
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()

        filename = f"{trace_name}_{impl_name}.json"
        prof.export_chrome_trace(filename)
        print(f"Exported Kineto trace: {filename}")


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--batch-sizes",
    default="10,100,1000,10000",
    help="Comma-separated list of batch sizes to benchmark",
)
@click.option(
    "--max-lengths",
    default="10,50,100,500",
    help="Comma-separated list of max lengths to benchmark",
)
@click.option(
    "--device",
    default="cuda",
    help="Device to run benchmark on (cpu or cuda)",
)
@click.option(
    "--iters",
    default=100,
    help="Number of iterations for timing",
)
@click.option(
    "--manual-seed/--skip-manual-seed",
    default=False,
    help="Use manual seed for reproduction.",
)
@click.option(
    "--export-trace/--no-export-trace",
    default=False,
    help="Export Kineto profiler traces for each configuration.",
)
def bench_repeat_arange(
    batch_sizes: str,
    max_lengths: str,
    device: str,
    iters: int,
    manual_seed: bool,
    export_trace: bool,
) -> None:
    """
    Run repeat_arange benchmarks across different configurations.
    """
    # set manual seed for reproducibility
    if manual_seed:
        torch.manual_seed(42)

    batch_size_list = [int(x) for x in batch_sizes.split(",")]
    max_length_list = [int(x) for x in max_lengths.split(",")]

    print("\n" + "=" * 70)
    print("REPEAT_ARANGE BENCHMARK")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Iterations: {iters}")
    print("=" * 70 + "\n")

    for batch_size in batch_size_list:
        for max_length in max_length_list:
            benchmark_repeat_arange(
                batch_size=batch_size,
                max_length=max_length,
                device=device,
                iters=iters,
                export_trace=export_trace,
            )


@cli.command()
@click.option(
    "--device",
    default="cuda",
    help="Device to run benchmark on (cpu or cuda)",
)
@click.option(
    "--manual-seed/--skip-manual-seed",
    default=False,
    help="Use manual seed for reproduction.",
)
@click.option(
    "--export-trace/--no-export-trace",
    default=False,
    help="Export Kineto profiler traces for each configuration.",
)
def bench_repeat_arange_quick(
    device: str, manual_seed: bool, export_trace: bool
) -> None:
    """
    Quick benchmark with representative configurations.
    """
    # set manual seed for reproducibility
    if manual_seed:
        torch.manual_seed(42)

    print("\n" + "=" * 70)
    print("REPEAT_ARANGE QUICK BENCHMARK")
    print("=" * 70 + "\n")

    # Representative configurations from real workloads
    configs = [
        (100, 50),  # Small: 100 sequences, avg length ~25
        (1000, 100),  # Medium: 1000 sequences, avg length ~50
        (10000, 100),  # Large: 10000 sequences, avg length ~50
        (1000, 500),  # Long sequences: 1000 sequences, avg length ~250
    ]

    for batch_size, max_length in configs:
        benchmark_repeat_arange(
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            iters=100,
            export_trace=export_trace,
        )


@cli.command()
@click.option(
    "--device",
    default="cuda",
    help="Device to run benchmark on (cpu or cuda)",
)
@click.option(
    "--manual-seed/--skip-manual-seed",
    default=False,
    help="Use manual seed for reproduction.",
)
@click.option(
    "--export-trace/--no-export-trace",
    default=False,
    help="Export Kineto profiler traces for each configuration.",
)
def bench_repeat_arange_scaling(
    device: str, manual_seed: bool, export_trace: bool
) -> None:
    """
    Benchmark scaling behavior with increasing batch sizes.
    """
    # set manual seed for reproducibility
    if manual_seed:
        torch.manual_seed(42)

    print("\n" + "=" * 70)
    print("REPEAT_ARANGE SCALING BENCHMARK")
    print("=" * 70 + "\n")

    # Test scaling with batch size
    print("Scaling with batch size (max_length=100):")
    for batch_size in [10, 100, 1000, 10000, 100000]:
        benchmark_repeat_arange(
            batch_size=batch_size,
            max_length=100,
            device=device,
            iters=50,
            export_trace=export_trace,
        )

    # Test scaling with sequence length
    print("\nScaling with sequence length (batch_size=1000):")
    for max_length in [10, 50, 100, 500, 1000]:
        benchmark_repeat_arange(
            batch_size=1000,
            max_length=max_length,
            device=device,
            iters=50,
            export_trace=export_trace,
        )


if __name__ == "__main__":
    cli()
