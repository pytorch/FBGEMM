# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any

import pandas as pd
import torch

from .quantize_ops import FP8RowwiseGroupedGemm


grouped_kernel_registry: list[str] = [
    "fp8_rowwise_grouped_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2",
    "fp8_rowwise_grouped_128x128x32x128_32x32_2x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2",
    "fp8_rowwise_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2",
    "fp8_rowwise_grouped_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2",
    "fp8_rowwise_grouped_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2",
    "fp8_rowwise_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2",
    "fp8_rowwise_grouped_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2",
    "fp8_rowwise_grouped_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2",
    "fp8_rowwise_grouped_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2",
    "fp8_rowwise_grouped_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2",
    "fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1",
    "fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3",
    "fp8_rowwise_grouped_256x128x128x64_32x32_2x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4",
    "fp8_rowwise_grouped_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3",
    "fp8_rowwise_grouped_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3",
    "fp8_rowwise_grouped_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3",
    "fp8_rowwise_grouped_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3",
    "fp8_rowwise_grouped_256x256x256x64_16x16_8x8_4x64x1_4x64x1_1x32x1x8_8x8x1_1x2_intrawave_v3",
    "fp8_rowwise_grouped_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4",
    "fp8_rowwise_grouped_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3",
    "fp8_rowwise_grouped_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2",
    "fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1",
    "fp8_rowwise_grouped_64x16x16x512_16x16_1x1_32x2x1_32x2x1_1x16x1x4_4x4x1_1x1_interwave_v2",
    "fp8_rowwise_grouped_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2",
    "fp8_rowwise_grouped_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_interwave_v2",
]


def main(args: Any):
    # Extract and format shape arguments.
    M = [int(m) for m in args.M.strip().split(",")]
    N = [int(n) for n in args.N.strip().split(",")]
    K = [int(k) for k in args.K.strip().split(",")]
    assert len(M) == len(N) == len(K), "M, N, and K must have the same length."

    # initialize tensors for benchmarking.
    A = []
    B = []
    num_groups = len(M)
    for i in range(num_groups):
        A.append(torch.randn(M[i], K[i], device="cuda"))
        B.append(torch.randn(N[i], K[i], device="cuda"))

    # Get quantized tensors.
    group_gemm_op = FP8RowwiseGroupedGemm()
    quantized_vals = group_gemm_op.quantize(A, B)
    # Iterate over kernels to find the most performant one.
    benchmark_results = []
    for kernel_name in grouped_kernel_registry:
        # Do a warmup run of the kernel.
        output = group_gemm_op.compute(*quantized_vals, kernel_name=kernel_name)
        # Benchmark this kernel implementation.
        ms_runtime = group_gemm_op.benchmark(
            *quantized_vals, use_cuda_graph=False, kernel_name=kernel_name
        )
        # Compute statistics for this kernel.
        tflops = 0
        gbps = 0
        for i in range(num_groups):
            tflops += 2 * M[i] * N[i] * K[i] / (ms_runtime / 1e3) / 1e12
            gbps += (
                (
                    quantized_vals[0][i].numel() * quantized_vals[0][i].element_size()
                    + quantized_vals[1][i].numel() * quantized_vals[1][i].element_size()
                    + output[i].numel() * output[i].element_size()
                )
                / (ms_runtime / 1e3)
                / 1e9
            )
        # Record results.
        benchmark_results.append(
            {
                "kernel_name": kernel_name,
                "ms_runtime": ms_runtime,
                "tflops": tflops,
                "gbps": gbps,
            }
        )
    # Print all results.
    print("Benchmark results:")
    for result in benchmark_results:
        print(f"Kernel: {result['kernel_name']}, TFLOPS: {result['tflops']}")
    # Report best kernel.
    best_kernel = min(benchmark_results, key=lambda x: x["ms_runtime"])
    print(f"Best kernel for this shape: {best_kernel['kernel_name']}")

    # If specified, save all results.
    if args.export_csv:
        df = pd.DataFrame(benchmark_results)
        df.to_csv("grouped_gemm_benchmark.csv", index=False)


def invoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Export results to a CSV file.",
    )
    parser.add_argument(
        "--M",
        required=True,
        help="Comma separated list of M values of each group to benchmark.",
    )
    parser.add_argument(
        "--N",
        required=True,
        help="Comma separated list of N values of each group to benchmark",
    )
    parser.add_argument(
        "--K",
        required=True,
        help="Comma separated list of K values of each group to benchmark.",
    )

    args = parser.parse_args()
    main(args)
