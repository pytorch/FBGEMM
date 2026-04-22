# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging

import click
import torch
from fbgemm_gpu.bench.bench_utils import benchmark_torch_function

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--flush-gpu-cache-size-mb", default=40)
@click.option("--iters", default=100)
@click.option("--batch-size", default=25)
@click.option("--m", default=2048)
@click.option("--n", default=100)
@click.option("--k", default=256)
@click.option("--num_warmups", default=2)
@click.option(
    "--manual-seed/--skip-manual-seed",
    default=False,
    help="Use manual seed for reproduction.",
)
@click.option("--device", default="cuda", help="Device type (default: cuda).")
@click.option(
    "--export-trace/--no-export-trace",
    default=False,
    help="Export Kineto trace to JSON file.",
)
def stride_gemm(
    flush_gpu_cache_size_mb: int,
    iters: int,
    batch_size: int,
    m: int,
    n: int,
    k: int,
    num_warmups: int,
    manual_seed: bool,
    device: str,
    export_trace: bool,
) -> None:
    # set manual seed for reproducibility
    if manual_seed:
        torch.manual_seed(42)

    A = torch.rand(m, batch_size, k, device=device).half()
    B = torch.rand(batch_size, k, n, device=device).half()
    bias = torch.rand(batch_size, n, device=device).half()
    bias_permute102 = bias.unsqueeze(1)

    # A100 40MB L2 cache
    elapse, _ = benchmark_torch_function(
        torch.ops.fbgemm.permute102_baddbmm_permute102,
        (bias, A, B),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        iters=iters,
        num_warmups=num_warmups,
    )
    logging.info(
        f"stride gemm fused: time: {elapse}, TFLOPS/sec: {2.0 * batch_size * m * n * k / elapse / 1.0e12: .2f}"
    )

    def ref_stride_gemm(
        bias_permute102: torch.Tensor, A: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        A_permute102 = A.permute(1, 0, 2)
        C_permute102 = torch.baddbmm(bias_permute102, A_permute102, B)
        C_ref = C_permute102.permute(1, 0, 2)  # (m, batch_size, n)
        return C_ref

    # A100 40MB L2 cache
    elapse_ref, _ = benchmark_torch_function(
        ref_stride_gemm,
        (bias_permute102, A, B),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        iters=iters,
        num_warmups=num_warmups,
    )
    logging.info(
        f"stride gemm unfused: time: {elapse_ref}, TFLOPS/sec: {2.0 * batch_size * m * n * k / elapse_ref / 1.0e12: .2f}"
    )

    if export_trace and device == "cuda":
        trace_name = f"stride_gemm_m{m}_b{batch_size}_n{n}_k{k}"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,  # pyre-ignore[16]
                torch.profiler.ProfilerActivity.CUDA,  # pyre-ignore[16]
            ],
            record_shapes=True,
        ) as prof:
            torch.ops.fbgemm.permute102_baddbmm_permute102(bias, A, B)
            ref_stride_gemm(bias_permute102, A, B)
        prof.export_chrome_trace(f"{trace_name}_trace.json")
        logging.info(f"Exported trace to {trace_name}_trace.json")


if __name__ == "__main__":
    cli()
