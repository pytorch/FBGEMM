# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random

import click
import hypothesis.strategies as st
import torch
from bench.benchmark_torch_function import (
    benchmark_torch_function,
)
from hypothesis import given, settings

logging.basicConfig(level=logging.DEBUG)

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--iters", default=100)
@click.option("--warmup-runs", default=2)
@settings(max_examples=10, deadline=None)
# pyre-ignore
@given(
    num_columns=st.sampled_from([2 ** n for n in range(4, 10)]),
    num_rows=st.sampled_from([2 ** n for n in range(4, 10)]),
)
def bench(
    flush_gpu_cache_size_mb: int,
    iters: int,
    num_columns: int,
    num_rows: int,
    warmup_runs: int,
) -> None:

    average_time = {
        "int8_quant": 0.0,
        "int4_quant": 0.0,
        "int2_quant": 0.0,
        "fp8_143_quant": 0.0,
        "fp8_152_quant": 0.0,
        "int8_dequant": 0.0,
        "int4_dequant": 0.0,
        "int2_dequant": 0.0,
        "fp8_143_dequant": 0.0,
        "fp8_152_dequant": 0.0,
    }

    input_data = torch.rand(num_rows, num_columns).float()
    quant_data_8bit = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_data)
    quant_data_4bit = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
        input_data, 4
    )
    quant_data_2bit = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
        input_data, 2
    )
    quant_data_fp8_143 = torch.ops.fbgemm.FloatToHFP8Quantized(
        input_data.contiguous(), 4, 3, 14, 2 ** (1 - 14 - 3), (2 - 2 ** (-3))
    )
    quant_data_fp8_152 = torch.ops.fbgemm.FloatToHFP8Quantized(
        input_data, 5, 2, 30, 2 ** (1 - 30 - 2), (2 - 2 ** (-2))
    )

    if torch.cuda.is_available():
        input_data = input_data.cuda()

    average_time["int8_quant"] = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized,
        input_data,
    )
    average_time["int4_quant"] = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        lambda input: torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input, 4),
        input_data,
    )

    average_time["int2_quant"] = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        lambda input: torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input, 2),
        input_data,
    )
    average_time["fp8_143_quant"] = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        lambda input: torch.ops.fbgemm.FloatToHFP8Quantized(
            input.contiguous(), 4, 3, 14, 2 ** (1 - 14 - 3), (2 - 2 ** (-3))
        ),
        input_data,
    )
    average_time["fp8_152_quant"] = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        lambda input: torch.ops.fbgemm.FloatToHFP8Quantized(
            input.contiguous(), 5, 2, 30, 2 ** (1 - 30 - 2), (2 - 2 ** (-2))
        ),
        input_data,
    )

    average_time["int8_dequant"] = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat,
        quant_data_8bit,
    )

    average_time["int4_dequant"] = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        lambda input: torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(input, 4),
        quant_data_4bit,
    )
    average_time["int2_dequant"] = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        lambda input: torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(input, 2),
        quant_data_2bit,
    )
    average_time["fp8_143_dequant"] = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        lambda input: torch.ops.fbgemm.HFP8QuantizedToFloat(input, 4, 3, 14),
        quant_data_fp8_143,
    )
    average_time["fp8_152_dequant"] = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        lambda input: torch.ops.fbgemm.HFP8QuantizedToFloat(input, 5, 2, 30),
        quant_data_fp8_152,
    )

    logging.info(f"-------------- ncols={num_columns}, nrows={num_rows}-------------")
    for k, t_time in average_time.items():
        logging.info(f"{k} time per iter: {t_time * 1.0e6:.0f}us")


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

    average_time_mixed_dim_fp32 = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatMixedDim,
        input_data,
        D_offsets,
        0,
    )  # output is FP32

    average_time_mixed_dim_fp16 = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatMixedDim,
        input_data,
        D_offsets,
        1,
    )  # output is FP16

    average_time_single_dim = benchmark_torch_function(
        flush_gpu_cache_size_mb,
        iters + warmup_runs,
        warmup_runs,
        torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat,
        input_data,
    )  # output is FP32

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
