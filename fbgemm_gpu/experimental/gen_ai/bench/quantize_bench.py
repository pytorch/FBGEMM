# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import torch

from .quantize_ops import get_quantize_ops, QuantizeOpBase


def set_amd_env_vars() -> None:
    print("Setting environment variables for AMD GPU performance")
    os.environ["DISABLE_ADDMM_HIP_LT"] = "0"
    os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
    os.environ["PYTORCH_TUNABLEOP_VERBOSE"] = "0"
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "hipblas_tuning_pt_llama.csv"
    os.environ["PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS"] = "30"
    os.environ["PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS"] = "30"


def benchmark(
    quantize_ops: List[QuantizeOpBase],
    m: int,
    n: int,
    k: int,
    kernels: Optional[List[str]] = None,
    bench_quantize: bool = False,
) -> Dict[str, Any]:
    # Create input tensors.
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    # Compute baseline output for correctness checking.
    out_ref = torch.matmul(A, B.t())
    # Keep track of results.
    results: Dict[str, Any] = {"M": m, "N": n, "K": k}
    # Benchmark each operator.
    for quantize_op in quantize_ops:
        # If kernel filter is provided, skip kernels that arent requested.
        kernel_requested = (kernels is None) or (
            kernels is not None and quantize_op.name in kernels
        )
        # Also check if the operator is supported.
        if kernel_requested and quantize_op.supported:
            # Get the quantized tensors for this operator.
            quantized_vals = quantize_op.quantize(A, B)
            # Compute the output given quantized values.
            output = quantize_op.compute(*quantized_vals)
            # Compare the quantize op output to reference as a sanity check.
            sim_check = torch.mean(torch.pow(torch.abs(output - out_ref), 2))

            # Now perform benchmark.
            if bench_quantize:
                # Benchmark both quantize and compute.
                ms_runtime = quantize_op.benchmark(A, B, bench_quantize=True)
            else:
                ms_runtime = quantize_op.benchmark(
                    *quantized_vals, bench_quantize=False
                )

            # Print out results for this op.
            tflops = 2 * m * n * k / (ms_runtime / 1e3) / 1e12
            print(f"{quantize_op.name} sim: {sim_check:.3f}.")
            print(f"{quantize_op.name} ms: {ms_runtime:.3f}.")
            print(f"{quantize_op.name} TFLOPS: {tflops:.3f}.")

            # Save results for this operator.
            results[f"{quantize_op.name}_sim"] = sim_check.item()
            results[f"{quantize_op.name}_ms"] = ms_runtime
            results[f"{quantize_op.name}_tflops"] = tflops

    return results


def plot_benchmark(results: List[Dict[str, Any]]) -> None:
    """Create a barplot visualizing the TFLOPS of each kernel."""
    # Reprocess into new dataframe with proper graph format.
    data = []
    # Extract measurements for each shape.
    for impl in results:
        mnk = f"{impl['M']}, {impl['N']}, {impl['K']}"
        # Iterate over keys to find tflops entries.
        for key in impl:
            if "tflops" in key:
                op_name = key.split("_tflops")[0]
                op_tflops = impl[key]
                data.append({"MNK": mnk, "kernel": op_name, "TFLOPS": op_tflops})

    # Create a barplot using seaborn.
    df = pd.DataFrame(data)
    plot = plt.figure()
    plt.xticks(rotation=30)
    plt.yscale("log")
    ax = sns.barplot(x="MNK", y="TFLOPS", hue="kernel", data=df)
    ax.tick_params(axis="x", labelsize=3)
    plot.savefig("quantize_ops_benchmark.png", dpi=300)


def main(args: Any):
    if args.enable_amd_env_vars:
        set_amd_env_vars()
    # Get operators to quantize.
    quantize_ops = get_quantize_ops()
    # If kernel filter is provided, parse it.
    if args.kernels is not None:
        kernels = args.kernels.strip().split(",")
    else:
        kernels = None
    # Enumerate shapes to benchmark.
    if args.M is None:
        M = [1, 4, 8, 16, 32, 64, 128, 2048, 4096, 8192, 16384]
    else:
        M = [int(m) for m in args.M.strip().split(",")]
    if args.N is None:
        N = [1280, 2304, 7168, 8192, 16384]
    else:
        N = [int(n) for n in args.N.strip().split(",")]
    if args.K is None:
        K = [1024, 3584, 8192, 16384]
    else:
        K = [int(k) for k in args.K.strip().split(",")]
    # List all shapes for simplicity.
    MNK = list(itertools.product(M, N, K))
    # Iterate over shapes and benchmark.
    benchmark_results = []
    for m, n, k in MNK:
        print(f"Benchmarking M={m}, N={n}, K={k}.")
        quantize_measurements = benchmark(
            quantize_ops, m, n, k, kernels, args.bench_quantize
        )
        benchmark_results.append(quantize_measurements)
    if args.export_csv:
        # Export results to a CSV file.
        df = pd.DataFrame(benchmark_results)
        df.to_csv("quantize_ops_benchmark.csv", index=False)
    if args.plot:
        plot_benchmark(benchmark_results)


def invoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Export results to a CSV file.",
    )
    parser.add_argument(
        "--plot",
        default=False,
        action="store_true",
        help="Create a plot of the benchmark measurements.",
    )
    parser.add_argument(
        "--enable_amd_env_vars",
        default=False,
        action="store_true",
        help="Enable a set of environment variables for AMD GPU performance",
    )
    parser.add_argument(
        "--bench_quantize",
        default=False,
        action="store_true",
        help="If set, include quantization cost in benchmark.",
    )
    parser.add_argument(
        "--kernels",
        default=None,
        help="Comma separated list of kernels to benchmark. Defaults to all kernels.",
    )
    parser.add_argument(
        "--M", default=None, help="Comma separated list of M values to benchmark."
    )
    parser.add_argument(
        "--N",
        default=None,
        help="Comma separated list of N values to benchmark",
    )
    parser.add_argument(
        "--K", default=None, help="Comma separated list of K values to benchmark."
    )

    args = parser.parse_args()
    main(args)
