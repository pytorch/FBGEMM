#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Run this benchmark with:
#   buck run @mode/opt deeplearning/fbgemm/fbgemm_gpu/examples:get_source_mask_benchmark

"""
Benchmark comparing CUDA kernel vs PyTorch reference implementation for get_source_mask.

This benchmark measures performance across various batch sizes and sequence lengths,
demonstrating the speedup achieved by the custom CUDA kernel.
"""

import time
from dataclasses import dataclass
from typing import List

import click
import torch
from fbgemm_gpu.sparse_ops import get_source_mask

torch.ops.load_library(
    "//deeplearning/fbgemm/fbgemm_gpu/src/jagged_tensor_ops:jagged_tensor_ops_gpu"
)


def get_source_mask_pytorch(
    num_sources: torch.Tensor, num_targets: torch.Tensor
) -> torch.Tensor:
    """Reference PyTorch implementation."""
    batch_size = num_sources.shape[0]
    device = num_sources.device
    skeleton = (
        torch.tensor([[True, False]], device=device).expand(batch_size, 2).flatten()
    )
    repeats = torch.stack([num_sources, num_targets], dim=1).flatten()
    return skeleton.repeat_interleave(repeats)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    batch_size: int
    max_sources: int
    max_targets: int
    cuda_time_ms: float
    pytorch_time_ms: float
    speedup: float
    total_elements: int


def benchmark_single_config(
    batch_size: int,
    max_sources: int,
    max_targets: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cuda",
) -> BenchmarkResult:
    """
    Benchmark a single configuration.

    Args:
        batch_size: Number of items in batch
        max_sources: Maximum number of sources per batch item
        max_targets: Maximum number of targets per batch item
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        device: Device to run on

    Returns:
        BenchmarkResult with timing information
    """
    torch_device = torch.device(device)

    num_sources = torch.randint(
        1, max_sources + 1, (batch_size,), dtype=torch.int64, device=torch_device
    )
    num_targets = torch.randint(
        1, max_targets + 1, (batch_size,), dtype=torch.int64, device=torch_device
    )

    total_elements = int((num_sources + num_targets).sum().item())

    for _ in range(num_warmup):
        _ = get_source_mask(num_sources, num_targets)
        _ = get_source_mask_pytorch(num_sources, num_targets)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = get_source_mask(num_sources, num_targets)
    torch.cuda.synchronize()
    cuda_time_ms = (time.perf_counter() - start_time) * 1000 / num_iterations

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = get_source_mask_pytorch(num_sources, num_targets)
    torch.cuda.synchronize()
    pytorch_time_ms = (time.perf_counter() - start_time) * 1000 / num_iterations

    speedup = pytorch_time_ms / cuda_time_ms if cuda_time_ms > 0 else 0.0

    return BenchmarkResult(
        batch_size=batch_size,
        max_sources=max_sources,
        max_targets=max_targets,
        cuda_time_ms=cuda_time_ms,
        pytorch_time_ms=pytorch_time_ms,
        speedup=speedup,
        total_elements=total_elements,
    )


@click.command()
@click.option(
    "--batch-sizes",
    type=str,
    default="32,64,128,256,512,1024",
    help="Comma-separated list of batch sizes to test",
)
@click.option(
    "--max-sources",
    type=str,
    default="10,50,100,500,2000,8000,20000",
    help="Comma-separated list of max source counts to test",
)
@click.option(
    "--max-targets",
    type=str,
    default="10,50,100,500,2000,8000,20000",
    help="Comma-separated list of max target counts to test",
)
@click.option(
    "--num-warmup",
    type=int,
    default=10,
    help="Number of warmup iterations",
)
@click.option(
    "--num-iterations",
    type=int,
    default=100,
    help="Number of timed iterations",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to run benchmarks on",
)
def cli(
    batch_sizes: str,
    max_sources: str,
    max_targets: str,
    num_warmup: int,
    num_iterations: int,
    device: str,
) -> None:
    """
    Benchmark get_source_mask CUDA kernel vs PyTorch reference.

    This tool measures the performance of the optimized CUDA kernel compared to
    a pure PyTorch implementation across various batch sizes and sequence lengths.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")

    print("=" * 100)
    print("FBGEMM get_source_mask CUDA Kernel Benchmark")
    print("=" * 100)
    print(f"\nDevice: {device}")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Timed iterations: {num_iterations}")
    print()

    batch_size_list = [int(x) for x in batch_sizes.split(",")]
    max_sources_list = [int(x) for x in max_sources.split(",")]
    max_targets_list = [int(x) for x in max_targets.split(",")]

    results: List[BenchmarkResult] = []

    print(
        f"{'Batch':>6} | {'MaxSrc':>6} | {'MaxTgt':>6} | {'TotalElems':>10} | "
        f"{'CUDA(ms)':>10} | {'PyTorch(ms)':>12} | {'Speedup':>8}"
    )
    print("-" * 100)

    for batch_size in batch_size_list:
        for max_src in max_sources_list:
            for max_tgt in max_targets_list:
                result = benchmark_single_config(
                    batch_size=batch_size,
                    max_sources=max_src,
                    max_targets=max_tgt,
                    num_warmup=num_warmup,
                    num_iterations=num_iterations,
                    device=device,
                )
                results.append(result)

                print(
                    f"{result.batch_size:6d} | {result.max_sources:6d} | "
                    f"{result.max_targets:6d} | {result.total_elements:10d} | "
                    f"{result.cuda_time_ms:10.4f} | {result.pytorch_time_ms:12.4f} | "
                    f"{result.speedup:8.2f}x"
                )

    print("\n" + "=" * 100)
    print("Summary Statistics")
    print("=" * 100)

    avg_speedup = sum(r.speedup for r in results) / len(results)
    max_speedup = max(r.speedup for r in results)
    min_speedup = min(r.speedup for r in results)

    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    print(f"Maximum Speedup: {max_speedup:.2f}x")
    print(f"Minimum Speedup: {min_speedup:.2f}x")

    fastest_config = max(results, key=lambda r: r.speedup)
    print("\nFastest Configuration:")
    print(
        f"  Batch Size: {fastest_config.batch_size}, "
        f"Max Sources: {fastest_config.max_sources}, "
        f"Max Targets: {fastest_config.max_targets}"
    )
    print(f"  Speedup: {fastest_config.speedup:.2f}x")
    print(f"  CUDA Time: {fastest_config.cuda_time_ms:.4f}ms")
    print(f"  PyTorch Time: {fastest_config.pytorch_time_ms:.4f}ms")


if __name__ == "__main__":
    cli()
