# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

import click
import torch
from fbgemm_gpu.bench.bench_utils import benchmark_torch_function

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
@click.option("--flush-gpu-cache-size-mb", default=40)
@click.option("--iters", default=100)
@click.option("--batch-size", default=25)
@click.option("--m", default=2048)
@click.option("--n", default=100)
@click.option("--k", default=256)
@click.option("--num_warmups", default=2)
def stride_gemm(
    flush_gpu_cache_size_mb: int,
    iters: int,
    batch_size: int,
    m: int,
    n: int,
    k: int,
    num_warmups: int,
) -> None:
    A = torch.rand(m, batch_size, k).half().cuda()
    B = torch.rand(batch_size, k, n).half().cuda()
    bias = torch.rand(batch_size, n).half().cuda()
    bias_permute102 = bias.unsqueeze(1)

    # A100 40MB L2 cache
    elapse, _ = benchmark_torch_function(
        torch.ops.fbgemm.permute102_baddbmm_permute102,
        (bias, A, B),
        flush_gpu_cache_size_mb,
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
        flush_gpu_cache_size_mb,
        iters=iters,
        num_warmups=num_warmups,
    )
    logging.info(
        f"stride gemm unfused: time: {elapse_ref}, TFLOPS/sec: {2.0 * batch_size * m * n * k / elapse_ref / 1.0e12: .2f}"
    )


if __name__ == "__main__":
    cli()
