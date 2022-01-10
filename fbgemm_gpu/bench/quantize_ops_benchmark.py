# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
import time
from typing import Callable, Tuple

import click
import torch
from torch import Tensor

logging.basicConfig(level=logging.DEBUG)

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


def benchmark_torch_function(
    func: Callable[[Tensor], Tensor],
    input: Tensor,
    flush_gpu_cache_size_mb: int,
) -> Tuple[float, Tensor]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # Flush the cache
        if flush_gpu_cache_size_mb:
            _ = torch.rand(
                flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float
            )
            torch.cuda.synchronize()
        start_event.record()
        # Benchmark code
        output = func(input)
        # Accumulate the time for iters iteration
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3
    else:
        start_time = time.time()
        output = func(input)
        elapsed_time = time.time() - start_time
    return float(elapsed_time), output


def benchmark_torch_mixdim_function(
    func: Callable[[Tensor, Tensor, int], Tensor],
    input: Tensor,
    D_offsets: Tensor,
    output_dtype: int,
    flush_gpu_cache_size_mb: int,
) -> Tuple[float, Tensor]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # Flush the cache
        if flush_gpu_cache_size_mb:
            _ = torch.rand(
                flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float
            )
            torch.cuda.synchronize()
        start_event.record()
        # Benchmark code
        output = func(input, D_offsets, output_dtype)
        # Accumulate the time for iters iteration
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3
    else:
        start_time = time.time()
        output = func(input, D_offsets, output_dtype)
        elapsed_time = time.time() - start_time
    return float(elapsed_time), output


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--iters", default=100)
@click.option("--num-columns", default=512)
@click.option("--num-rows", default=512)
@click.option("--warmup-runs", default=2)
def bench(
    flush_gpu_cache_size_mb: int,
    iters: int,
    num_columns: int,
    num_rows: int,
    warmup_runs: int,
) -> None:

    total_time = {
        "8bit_quant": 0.0,
        "4bit_quant": 0.0,
        "2bit_quant": 0.0,
        "8bit_dequant": 0.0,
        "4bit_dequant": 0.0,
        "2bit_dequant": 0.0,
    }

    input_data = torch.rand(num_rows, num_columns).float()
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    for step in range(iters + warmup_runs):
        time, quant_data_8bit = benchmark_torch_function(
            torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized,
            input_data,
            flush_gpu_cache_size_mb,
        )
        if step >= warmup_runs:
            total_time["8bit_quant"] += time

        time, quant_data_4bit = benchmark_torch_function(
            lambda input: torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                input, 4
            ),
            input_data,
            flush_gpu_cache_size_mb,
        )
        if step >= warmup_runs:
            total_time["4bit_quant"] += time

        time, quant_data_2bit = benchmark_torch_function(
            lambda input: torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                input, 2
            ),
            input_data,
            flush_gpu_cache_size_mb,
        )
        if step >= warmup_runs:
            total_time["2bit_quant"] += time

        time, _ = benchmark_torch_function(
            torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat,
            quant_data_8bit,
            flush_gpu_cache_size_mb,
        )
        if step >= warmup_runs:
            total_time["8bit_dequant"] += time

        time, _ = benchmark_torch_function(
            lambda input: torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(
                input, 4
            ),
            quant_data_4bit,
            flush_gpu_cache_size_mb,
        )
        if step >= warmup_runs:
            total_time["4bit_dequant"] += time

        time, _ = benchmark_torch_function(
            lambda input: torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(
                input, 2
            ),
            quant_data_2bit,
            flush_gpu_cache_size_mb,
        )
        if step >= warmup_runs:
            total_time["2bit_dequant"] += time

    logging.info(f"-------------- ncols={num_columns}, nrows={num_rows}-------------")
    for k, t_time in total_time.items():
        logging.info(f"{k} time per iter: {t_time / iters * 1.0e6:.0f}us")


@cli.command()
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--iters", default=100)
@click.option("--batch_size", default=512)
@click.option("--num_tables", default=256)
@click.option("--min_dim", default=1)
@click.option("--max_dim", default=128)
@click.option("--warmup-runs", default=2)
def mixdim(
    flush_gpu_cache_size_mb: int,
    iters: int,
    batch_size: int,
    num_tables: int,
    min_dim: int,
    max_dim: int,
    warmup_runs: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    random.seed(0)
    table_dims = [
        random.randint(min_dim, max_dim) * 8 for _ in range(num_tables)
    ]  # assume table dimensions are multiples of 8
    table_dims_with_qparams = [d + 8 for d in table_dims]
    D_offsets = (
        torch.cumsum(torch.tensor([0] + table_dims_with_qparams), dim=0)
        .to(torch.int)
        .cuda()
    )
    input_refs = [torch.randn((batch_size, d)).cuda() for d in table_dims]
    input_refs_int8 = [
        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(t) for t in input_refs
    ]
    input_data = torch.concat(input_refs_int8, dim=1).contiguous()
    total_time_mixed_dim_fp32 = 0.0
    total_time_mixed_dim_fp16 = 0.0
    total_time_single_dim = 0.0

    for step in range(iters + warmup_runs):
        time, _ = benchmark_torch_mixdim_function(
            torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatMixedDim,
            input_data,
            D_offsets,
            0,
            0,
        )  # output is FP32
        if step >= warmup_runs:
            total_time_mixed_dim_fp32 += time

        time, _ = benchmark_torch_mixdim_function(
            torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatMixedDim,
            input_data,
            D_offsets,
            1,
            0,
        )  # output is FP16
        if step >= warmup_runs:
            total_time_mixed_dim_fp16 += time

        time, _ = benchmark_torch_function(
            torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat,
            input_data,
            0,
        )  # output is FP32
        if step >= warmup_runs:
            total_time_single_dim += time

    average_time_mixed_dim_fp32 = total_time_mixed_dim_fp32 / iters
    average_time_mixed_dim_fp16 = total_time_mixed_dim_fp16 / iters
    average_time_single_dim = total_time_single_dim / iters

    print(
        f"Input tensor batch_size: {batch_size}, num_tables: {num_tables}, tensor_size: {input_data.numel() / (1 << 30)} GB, average table dimension: {sum(table_dims) * 1.0/num_tables}."
    )
    print(
        f"Mixed dim dequantize average time per iter FP32: {average_time_mixed_dim_fp32} s, bandwidth : {input_data.numel() / (1 << 30) / average_time_mixed_dim_fp32} GB/s."
    )
    print(
        f"Mixed dim dequantize average time per iter FP16: {average_time_mixed_dim_fp16} s, bandwidth : {input_data.numel() / (1 << 30) / average_time_mixed_dim_fp16} GB/s."
    )
    print(
        f"Single dim dequantize average time per iter FP32: {average_time_single_dim} s, bandwidth: {input_data.numel() / (1 << 30) / average_time_single_dim} GB/s."
    )


if __name__ == "__main__":
    cli()
