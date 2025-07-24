# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools

import os

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
import torch

try:
    from accelerators.utils.torch_profiler import profiler_or_nullcontext
except ImportError:
    from contextlib import nullcontext

    class profiler_or_nullcontext(nullcontext):
        def __init__(self, *args, **kwargs):
            super().__init__()


from fbgemm_gpu.experimental.gen_ai.bench.quantize_ops import (
    get_quantize_ops,
    QuantizeOpBase,
)


def generate_group_tensor(G, M):
    """
    Generate a tensor with G elements whose integer elements sum to A.

    Args:
        G (int): Number of elements in the tensor.
        M (int): Sum of the elements in the tensor.

    Returns:
        torch.Tensor: A tensor with G elements whose integer elements sum to M.
    """

    # First, we generate a random tensor with G elements
    random_tensor = torch.rand(G)
    # Then, we normalize this tensor so it sums up to 1
    normalized_tensor = random_tensor / random_tensor.sum()
    # Finally, we multiply this tensor by M and round to the nearest integer
    output_tensor = torch.round(normalized_tensor * M).to(torch.int64)
    # Adjust the last element to ensure the sum is exactly M
    output_tensor[-1] += max(0, M - output_tensor.sum())
    return output_tensor.tolist()


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


def get_llama_shapes() -> List[Tuple[int, int, int, int]]:
    # Helper function that returns a list of shapes relevant to llama.

    llama_shapes = []
    for M in [1, 16, 32, 64, 96, 128, 16384]:
        # Add shapes for llama3 70B
        llama_shapes += [
            (1, M, 1280, 8192),
            (1, M, 8192, 1024),
            (1, M, 7168, 8192),
            (1, M, 8192, 3584),
        ]
        # Add shapes for llama3 405B
        llama_shapes += [
            (1, M, 13312, 6656),
            (1, M, 13312, 16384),
            (1, M, 16384, 6656),
            (1, M, 16384, 16384),
        ]
        # Add shapes for llama4 Scout/Maverick (17Bx{16,128})
        llama_shapes += [
            (1, M, 896, 5120),
            (1, M, 5120, 640),
            (1, M, 2048, 5120),
            (1, M, 5120, 1024),
        ]

    return llama_shapes


def get_ldm_shapes() -> List[Tuple[int, int, int, int]]:
    # Helper function that returns a list of shapes relevant to ldm.
    return [
        (1, 1536, 3584, 3584),
        (1, 8192, 9728, 3584),
        (1, 8192, 3584, 9728),
        (1, 8192, 3584, 3584),
        (1, 4096, 3584, 3584),
        (1, 768, 3584, 3584),
        (1, 4096, 9728, 3584),
        (1, 4096, 3584, 9728),
        (1, 7200, 3584, 3584),
        (1, 7200, 9728, 3584),
        (1, 7200, 3584, 9728),
        (1, 3600, 3584, 3584),
        (1, 3600, 9728, 3584),
        (1, 3600, 3584, 9728),
        (1, 1536, 4096, 4096),
        (1, 3600, 4096, 4096),
        (1, 3600, 11008, 4096),
        (1, 3600, 4096, 11008),
        (1, 4096, 4096, 4096),
        (1, 4096, 11008, 4096),
        (1, 4096, 4096, 11008),
        (1, 32768, 128, 8192),
        (1, 32768, 8192, 1024),
        (1, 32768, 8192, 3072),
        (1, 32768, 3072, 8192),
        (1, 32768, 1024, 8192),
    ]


@dataclass
class Metrics:
    op_name: str

    sim: float = 0.0
    ms: float = 0.0
    tflops: float = 0.0
    gbps: float = 0.0

    def __str__(self) -> str:
        return (
            "%s sim: %.3f.\n%s ms: %.3f. \n" "%s TFLOPS: %.3f. \n%s GB/s: %.3f."
        ) % (
            self.op_name,
            self.sim,
            self.op_name,
            self.ms,
            self.op_name,
            self.tflops,
            self.op_name,
            self.gbps,
        )


def benchmark_grouped(
    quantize_ops: List[QuantizeOpBase],
    b: List[int],
    m: List[int],
    n: List[int],
    k: List[int],
    bench_quantize: bool = False,
    use_rotating_buffer_bench: bool = False,
    use_cuda_graph: bool = True,
    trace: bool = False,
    num_iters: int = 1,
    fast_accum: bool = True,
    torch_compile: bool = False,
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
        metrics = Metrics(op_name=quantize_op.name)
        # Set fast accum mode if applicable.
        if hasattr(quantize_op, "fast_accum"):
            quantize_op.fast_accum = fast_accum
        if hasattr(quantize_op, "torch_compile"):
            quantize_op.torch_compile = torch_compile
        # Get the quantized tensors for this operator.
        preprocessed_args = quantize_op.preprocess(A, B)
        quantized_vals = quantize_op.quantize(*preprocessed_args)
        # Compute the output given quantized values.
        output = quantize_op.compute(*quantized_vals)
        # Some kernels may pad output, just take the first m values of each row.
        if isinstance(output, torch.Tensor) and output.ndim == 2:
            # Output is stacked and needs to be split.
            output = torch.split(output, m, dim=0)
        else:
            # Otherwise output may be padded or require unbinding.
            output = [o[: m[i]] for i, o in enumerate(output)]
        # Compare the quantize op output to reference as a sanity check.
        for i in range(num_groups):
            if m[i] > 0:
                metrics.sim += float(
                    torch.mean(torch.pow(output[i] - out_ref[i], 2)).item()
                )
        for _ in range(num_iters):
            # Now perform benchmark.
            if bench_quantize:
                # Benchmark both quantize and compute.
                with profiler_or_nullcontext(enabled=trace, with_stack=True):
                    ms_runtime = quantize_op.benchmark(
                        *preprocessed_args,
                        bench_quantize=True,
                        use_rotating_buffer_bench=use_rotating_buffer_bench,
                        use_cuda_graph=use_cuda_graph,
                    )
            else:
                with profiler_or_nullcontext(enabled=trace, with_stack=True):
                    ms_runtime = quantize_op.benchmark(
                        *quantized_vals,
                        bench_quantize=False,
                        use_rotating_buffer_bench=use_rotating_buffer_bench,
                        use_cuda_graph=use_cuda_graph,
                    )

            # Print out results for this op.
            for i in range(num_groups):
                metrics.tflops += (
                    2 * b[i] * m[i] * n[i] * k[i] / (ms_runtime / 1e3) / 1e12
                )
                output_multiplier = 2 if "fuse_scatter_add" in quantize_op.name else 1
                if m[i] > 0:
                    metrics.gbps += (
                        (
                            b[i] * m[i] * k[i] * quantized_vals[0][0].element_size()
                            + b[i] * n[i] * k[i] * quantized_vals[1][0].element_size()
                            + output_multiplier
                            * b[i]
                            * m[i]
                            * n[i]
                            * output[0].element_size()
                        )
                        / (ms_runtime / 1e3)
                        / 1e9
                    )
            metrics.ms += ms_runtime
        metrics.ms /= num_iters
        metrics.tflops /= num_iters
        metrics.gbps /= num_iters
        print(f"Average metrics over {num_iters} iterations: \n{metrics}")

        # Save results for this operator.
        results[f"{quantize_op.name}_sim"] = metrics.sim
        results[f"{quantize_op.name}_ms"] = metrics.ms
        results[f"{quantize_op.name}_tflops"] = metrics.tflops
        results[f"{quantize_op.name}_gb/s"] = metrics.gbps

    return results


def benchmark(
    quantize_ops: List[QuantizeOpBase],
    b: int,
    m: int,
    n: int,
    k: int,
    bench_quantize: bool = False,
    use_rotating_buffer_bench: bool = False,
    use_cuda_graph: bool = True,
    trace: bool = False,
    num_iters: int = 1,
    fast_accum: bool = True,
    torch_compile: bool = False,
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
        metrics = Metrics(op_name=quantize_op.name)
        # Set fast accum mode if applicable.
        if hasattr(quantize_op, "fast_accum"):
            quantize_op.fast_accum = fast_accum
        if hasattr(quantize_op, "torch_compile"):
            quantize_op.torch_compile = torch_compile
        # Preprocess data if needed.
        preprocessed_args = quantize_op.preprocess(A, B)
        # Get the quantized tensors for this operator.
        quantized_vals = quantize_op.quantize(*preprocessed_args)
        # Compute the output given quantized values.
        output = quantize_op.compute(*quantized_vals)
        # Compare the quantize op output to reference as a sanity check.
        # TODO(shikaili): This calculation is incorrect for scatter add fusion.
        metrics.sim = torch.mean(torch.pow(output - out_ref, 2)).item()

        for _ in range(num_iters):
            # Now perform benchmark.
            if bench_quantize:
                # Benchmark both quantize and compute.
                with profiler_or_nullcontext(enabled=trace, with_stack=True):
                    ms_runtime = quantize_op.benchmark(
                        *preprocessed_args,
                        bench_quantize=True,
                        use_rotating_buffer_bench=use_rotating_buffer_bench,
                        use_cuda_graph=use_cuda_graph,
                    )
            else:
                with profiler_or_nullcontext(enabled=trace, with_stack=True):
                    ms_runtime = quantize_op.benchmark(
                        *quantized_vals,
                        bench_quantize=False,
                        use_rotating_buffer_bench=use_rotating_buffer_bench,
                        use_cuda_graph=use_cuda_graph,
                    )

            # Print out results for this op.
            metrics.tflops += 2 * b * m * n * k / (ms_runtime / 1e3) / 1e12
            metrics.gbps += (
                (
                    quantized_vals[0].numel() * quantized_vals[0].element_size()
                    + quantized_vals[1].numel() * quantized_vals[1].element_size()
                    + output.numel() * output.element_size()
                )
                / (ms_runtime / 1e3)
                / 1e9
            )
            metrics.ms += ms_runtime
        # Print out results for this op.
        metrics.ms /= num_iters
        metrics.tflops /= num_iters
        metrics.gbps /= num_iters
        print(f"Average metrics over {num_iters} iterations: \n{metrics}")

        # Save results for this operator.
        results[f"{quantize_op.name}_sim"] = metrics.sim
        results[f"{quantize_op.name}_ms"] = metrics.ms
        results[f"{quantize_op.name}_tflops"] = metrics.tflops
        results[f"{quantize_op.name}_gb/s"] = metrics.gbps

    return results


def plot_benchmark(results: List[Dict[str, Any]], output_dir: str) -> None:
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
    img_fn = os.path.join(output_dir, "quantize_ops_benchmark.png")
    plot.savefig(img_fn, dpi=300)
    print(f"Plot saved to {img_fn}")


def collect_kernels_to_profile(kernels: Optional[List[str]]) -> List[QuantizeOpBase]:
    # Get existing quantization operators.
    quantize_ops = get_quantize_ops()
    quantize_ops = [op for op in quantize_ops if op.supported]
    if kernels is None:
        return quantize_ops
    return [op for op in quantize_ops if op.name in kernels]


def main(args: Any):
    if args.enable_amd_env_vars:
        set_amd_env_vars()
    # If kernel filter is provided, parse it. Else, benchmark all kernels.
    quantize_ops = collect_kernels_to_profile(
        args.kernels.strip().split(",") if args.kernels else None
    )

    if len(quantize_ops) == 0:
        raise Exception("No valid kernels to benchmark.")

    if args.num_iters < 1:
        print("Number of iterations must be at least 1.")
        args.num_iters = 1

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
        elif args.use_ldm_shapes:
            MNK = get_ldm_shapes()
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
            if args.pair_NK:
                if len(N) != len(K):
                    raise Exception("N and K must be the same length in pair_NK mode.")
                NK = zip(N, K)
                MNK = list(
                    (B, M, N, K) for (B, M, (N, K)) in itertools.product(B, M, NK)
                )
            else:
                MNK = list(itertools.product(B, M, N, K))
    # When groups is provided transform shapes into grouped format.
    if args.groups:
        groups = [int(g) for g in args.groups.strip().split(",")]
        if args.total_M:
            MNK = [
                [
                    [b] * g,
                    generate_group_tensor(g, int(args.total_M)),
                    [n] * g,
                    [k] * g,
                ]
                for g in groups
                for b, _, n, k in MNK
            ]
        else:
            MNK = [
                [[b] * g, [m] * g, [n] * g, [k] * g]
                for g in groups
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
            args.bench_quantize,
            args.use_rotating_buffer_bench,
            not args.no_cuda_graph,
            args.trace,
            args.num_iters,
            not args.disable_fast_accum,
            args.torch_compile,
        )
        benchmark_results.append(quantize_measurements)
    if args.export_csv or args.plot:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.export_csv:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(
            args.output_dir, f"quantize_ops_benchmark_{datetime_str}.csv"
        )
        print(f"CSV saved to {csv_file}")
        # Export results to a CSV file.
        df = pd.DataFrame(benchmark_results)
        df.to_csv(csv_file, index=False)
    if args.plot:
        plot_benchmark(benchmark_results, args.output_dir)


def invoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="/tmp", help="Directory to save plots and csvs to"
    )
    parser.add_argument(
        "--num_iters",
        default=1,
        type=int,
        help="Number of iterations to repeat each benchmark.",
    )
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
        "--pair_NK",
        default=False,
        action="store_true",
        help="If set, instead of benchmarking cartesian product of N * K, benchmark consecutive NK pairs together.",
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
        help="If set with grouped mode, repeat input shapes this many times. Comma separated list of groups to benchmark",
    )
    parser.add_argument(
        "--total_M",
        default=None,
        help="If set, Adjusts the M values to sum to this number. "
        "This can help simulate real grouped workloads.",
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
    parser.add_argument(
        "--use_ldm_shapes",
        default=False,
        action="store_true",
        help="If set, benchmark using fixed shapes relevant to ldm workloads.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action="store_true",
        help="If set, produce a performance trace of the benchmark.",
    )
    parser.add_argument(
        "--disable_fast_accum",
        default=False,
        action="store_true",
        help="If set, disable fast accumulation for FP8 implementations.",
    )
    parser.add_argument(
        "--torch_compile",
        default=False,
        action="store_true",
        help="If set, torch.compile will be used for scaled_mm backed ops.",
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
