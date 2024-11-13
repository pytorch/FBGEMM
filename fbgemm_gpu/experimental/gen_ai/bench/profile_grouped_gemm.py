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
    for kernel_name in torch.ops.fbgemm.get_f8f8bf16_rowwise_grouped_kernels():
        # Do a warmup run of the kernel.
        output = group_gemm_op.compute(*quantized_vals, kernel_name=kernel_name)
        # Benchmark this kernel implementation.
        ms_runtime = group_gemm_op.benchmark(
            *quantized_vals, use_cuda_graph=True, kernel_name=kernel_name
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
        print(f"Kernel: {kernel_name}, ms: {ms_runtime:.4f}, TFLOPS: {tflops:.2f}")
        benchmark_results.append(
            {
                "kernel_name": kernel_name,
                "ms_runtime": ms_runtime,
                "tflops": tflops,
                "gbps": gbps,
            }
        )
    # Report best kernel.
    best_kernel = min(benchmark_results, key=lambda x: x["ms_runtime"])
    print(
        f"Best kernel for this shape: {best_kernel['kernel_name']}: {best_kernel['tflops']:.2f} TFLOPS"
    )

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
