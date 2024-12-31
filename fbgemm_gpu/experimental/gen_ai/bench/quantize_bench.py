# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

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


def get_llama_shapes() -> List[Tuple[int, int, int]]:
    # Helper function that returns a list of shapes relevant to llama.

    llama_shapes = []
    for M in [1, 16, 32, 64, 96, 128, 16384]:
        # Add shapes for llama 70B
        llama_shapes += [
            (M, 1280, 8192),
            (M, 8192, 1024),
            (M, 7168, 8192),
            (M, 8192, 3584),
        ]
        # Add shapes for llama 405B
        llama_shapes += [
            (M, 13312, 6656),
            (M, 13312, 16384),
            (M, 16384, 6656),
            (M, 16384, 16384),
        ]

    return llama_shapes


def benchmark_grouped(
    quantize_ops: List[QuantizeOpBase],
    b: List[int],
    m: List[int],
    n: List[int],
    k: List[int],
    kernels: Optional[List[str]] = None,
    bench_quantize: bool = False,
    use_rotating_buffer_bench: bool = False,
    use_cuda_graph: bool = True,
) -> Dict[str, Any]:
    num_groups = len(m)
    # Create input tensors.
    A = []
    B = []
    for i in range(num_groups):
        if b[i] > 1:
            A.append(torch.randn(b[i], m[i], k[i], device="cuda", dtype=torch.bfloat16))
            B.append(torch.randn(b[i], n[i], k[i], device="cuda", dtype=torch.bfloat16))
        else:
            A.append(torch.randn(m[i], k[i], device="cuda", dtype=torch.bfloat16))
            B.append(torch.randn(n[i], k[i], device="cuda", dtype=torch.bfloat16))
    # Compute baseline output for correctness checking.
    out_ref = []
    for i in range(num_groups):
        out_ref.append(torch.matmul(A[i], B[i].t()))
    # Keep track of results.
    # Only log all shapes in a group if they are unique.
    log_m = m[0] if len(np.unique(m)) == 1 else m
    log_n = n[0] if len(np.unique(n)) == 1 else n
    log_k = k[0] if len(np.unique(k)) == 1 else k
    results: Dict[str, Any] = {"M": log_m, "N": log_n, "K": log_k, "groups": num_groups}
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
            # Some kernels may pad output, just take the first m values of each row.
            output = [o[: m[i]] for i, o in enumerate(output)]
            # Compare the quantize op output to reference as a sanity check.
            sim_check: float = 0
            for i in range(num_groups):
                sim_check += float(
                    torch.mean(torch.pow(output[i] - out_ref[i], 2)).item()
                )

            # Now perform benchmark.
            if bench_quantize:
                # Benchmark both quantize and compute.
                ms_runtime = quantize_op.benchmark(
                    A,
                    B,
                    bench_quantize=True,
                    use_rotating_buffer_bench=use_rotating_buffer_bench,
                    use_cuda_graph=use_cuda_graph,
                )
            else:
                ms_runtime = quantize_op.benchmark(
                    *quantized_vals,
                    bench_quantize=False,
                    use_rotating_buffer_bench=use_rotating_buffer_bench,
                    use_cuda_graph=use_cuda_graph,
                )

            # Print out results for this op.
            tflops = 0
            gbps = 0
            for i in range(num_groups):
                tflops += 2 * b[i] * m[i] * n[i] * k[i] / (ms_runtime / 1e3) / 1e12
                gbps += (
                    (
                        quantized_vals[0][i][: m[i]].numel()
                        * quantized_vals[0][i][: m[i]].element_size()
                        + quantized_vals[1][i].numel()
                        * quantized_vals[1][i].element_size()
                        + output[i].numel() * output[i].element_size()
                    )
                    / (ms_runtime / 1e3)
                    / 1e9
                )
            print(f"{quantize_op.name} sim: {sim_check:.3f}.")
            print(f"{quantize_op.name} ms: {ms_runtime:.3f}.")
            print(f"{quantize_op.name} TFLOPS: {tflops:.3f}.")
            print(f"{quantize_op.name} GB/s: {gbps:.3f}.")

            # Save results for this operator.
            results[f"{quantize_op.name}_sim"] = sim_check
            results[f"{quantize_op.name}_ms"] = ms_runtime
            results[f"{quantize_op.name}_tflops"] = tflops
            results[f"{quantize_op.name}_gb/s"] = gbps

    return results


def benchmark(
    quantize_ops: List[QuantizeOpBase],
    b: int,
    m: int,
    n: int,
    k: int,
    kernels: Optional[List[str]] = None,
    bench_quantize: bool = False,
    use_rotating_buffer_bench: bool = False,
    use_cuda_graph: bool = True,
) -> Dict[str, Any]:
    # Create input tensors.
    if b > 1:
        A = torch.randn(b, m, k, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(b, n, k, device="cuda", dtype=torch.bfloat16)
    else:
        A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

    # Compute baseline output for correctness checking.
    out_ref = torch.matmul(A, torch.transpose(B, -2, -1))
    # Keep track of results.
    results: Dict[str, Any] = {"B": b, "M": m, "N": n, "K": k}
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
            sim_check = torch.mean(torch.pow(output - out_ref, 2))

            # Now perform benchmark.
            if bench_quantize:
                # Benchmark both quantize and compute.
                ms_runtime = quantize_op.benchmark(
                    A,
                    B,
                    bench_quantize=True,
                    use_rotating_buffer_bench=use_rotating_buffer_bench,
                    use_cuda_graph=use_cuda_graph,
                )
            else:
                ms_runtime = quantize_op.benchmark(
                    *quantized_vals,
                    bench_quantize=False,
                    use_rotating_buffer_bench=use_rotating_buffer_bench,
                    use_cuda_graph=use_cuda_graph,
                )

            # Print out results for this op.
            tflops = 2 * b * m * n * k / (ms_runtime / 1e3) / 1e12
            gbps = (
                (
                    quantized_vals[0].numel() * quantized_vals[0].element_size()
                    + quantized_vals[1].numel() * quantized_vals[1].element_size()
                    + output.numel() * output.element_size()
                )
                / (ms_runtime / 1e3)
                / 1e9
            )
            print(f"{quantize_op.name} sim: {sim_check:.3f}.")
            print(f"{quantize_op.name} ms: {ms_runtime:.3f}.")
            print(f"{quantize_op.name} TFLOPS: {tflops:.3f}.")
            print(f"{quantize_op.name} GB/s: {gbps:.3f}.")

            # Save results for this operator.
            results[f"{quantize_op.name}_sim"] = sim_check.item()
            results[f"{quantize_op.name}_ms"] = ms_runtime
            results[f"{quantize_op.name}_tflops"] = tflops
            results[f"{quantize_op.name}_gb/s"] = gbps

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
    if args.grouped and not args.groups:
        # In grouped mode, M, N, and K represent the groups of a single gemm.
        assert args.M is not None and args.N is not None and args.K is not None
        M = [int(m) for m in args.M.strip().split(",")]
        N = [int(n) for n in args.N.strip().split(",")]
        K = [int(k) for k in args.K.strip().split(",")]
        if args.B is None:
            B = [1] * len(M)
        else:
            B = [int(b) for b in args.B.strip().split(",")]
        assert (
            len(M) == len(N) == len(K) == len(B)
        ), "B, M, N, and K must be the same length in grouped mode."

        # Note this is a single grouped gemm.
        MNK = [[B, M, N, K]]
    else:
        if args.B is None:
            B = [1]
        else:
            B = [int(b) for b in args.B.strip().split(",")]
        if args.use_llama_shapes:
            MNK = get_llama_shapes()
        else:
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
            MNK = list(itertools.product(B, M, N, K))
    # When groups is provided transform shapes into grouped format.
    if args.groups:
        groups = int(args.groups)
        MNK = [
            [[b] * groups, [m] * groups, [n] * groups, [k] * groups]
            for b, m, n, k in MNK
        ]

    # Iterate over shapes and benchmark.
    benchmark_results = []
    for b, m, n, k in MNK:
        print(f"Benchmarking B={b}, M={m}, N={n}, K={k}.")
        benchmark_func = benchmark_grouped if args.grouped else benchmark
        quantize_measurements = benchmark_func(
            quantize_ops,
            b,  # pyre-ignore[6]: Incompatible parameter type [6]
            m,  # pyre-ignore[6]: Incompatible parameter type [6]
            n,  # pyre-ignore[6]: Incompatible parameter type [6]
            k,  # pyre-ignore[6]: Incompatible parameter type [6]
            kernels,
            args.bench_quantize,
            args.use_rotating_buffer_bench,
            not args.no_cuda_graph,
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
        "--B", default=None, help="Comma separated list of batches to benchmark."
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
    parser.add_argument(
        "--grouped",
        default=False,
        action="store_true",
        help="If set, do grouped gemm. In this mode, M, N, and K are interpreted "
        "as the size of groups. The length of each must be the same.",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="If set with grouped mode, repeat input shapes this many times.",
    )
    parser.add_argument(
        "--no_cuda_graph",
        default=False,
        action="store_true",
        help="If set, do not use cuda graph for benchmarking.",
    )
    parser.add_argument(
        "--use_rotating_buffer_bench",
        default=False,
        action="store_true",
        help="If set, use rotating buffer to benchmark.",
    )
    parser.add_argument(
        "--use_llama_shapes",
        default=False,
        action="store_true",
        help="If set, benchmark using fixed shapes relevant to llama workloads.",
    )

    args = parser.parse_args()
    main(args)
