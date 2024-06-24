# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, Tuple

import click

import torch
import triton  # @manual

from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
    matmul_fp8_block,
    matmul_fp8_row,
    quantize_fp8_block,
    quantize_fp8_row,
)
from torch._tensor import Tensor


@click.command()
@click.option("--cuda-graph", type=bool, default=True)
@click.option("--rowwise-tma", is_flag=True, default=False)
def bench(cuda_graph: bool, rowwise_tma: bool) -> None:
    """Benchmark bf16 vs scale/cast + fp8."""

    def _run_benchmark(
        bench_factory: Callable[
            [torch.Tensor, torch.Tensor], Callable[[], torch.Tensor]
        ],
        shape: Tuple[int, int, int] = (1024, 1024, 1024),
        tag: str = "",
    ) -> None:
        # Benchmarks the function returned by bench_factory.
        # Any pre-processing that should not be benchmarked can occur inside bench_factory.
        m, n, k = shape

        input_shape = (m, k)
        weight_shape = (n, k)

        base_dtype = torch.bfloat16
        input_ = torch.randn(input_shape, device="cuda", dtype=base_dtype)
        weight_ = torch.randn(weight_shape, device="cuda", dtype=base_dtype)

        gemm_fn = bench_factory(input_, weight_)

        if cuda_graph:
            bench_stream = torch.cuda.Stream()
            with torch.cuda.stream(bench_stream):
                ms = triton.testing.do_bench_cudagraph(
                    lambda: gemm_fn(),
                    rep=100,
                )
        else:
            ms = triton.testing.do_bench(
                lambda: gemm_fn(),
                warmup=25,
                rep=100,
            )

        tflops = (2 * m * n * k) / 1e12
        sec = ms / 1e3
        perf_str = f"{tflops / sec:.2f}"
        print(
            f"{(tag + ':').ljust(40)}\tshape {str(shape):<25} tflops {perf_str:<8} ms {ms:.3f}"
        )

    shapes = [
        (8192, 8192, 512),
        (8192, 8192, 8192),
        (65536, 8192, 7168),
        (65536, 3584, 8192),
        (8192, 14336, 4096),
    ]
    for shape in shapes:
        _run_benchmark(bf16_bench, shape=shape, tag="bf16")
        _run_benchmark(scale_row_bench, shape=shape, tag="fp8 scale + row gemm")
        _run_benchmark(scale_block_bench, shape=shape, tag="fp8 scale + block gemm")
        _run_benchmark(
            row_gemm_bench,
            shape=shape,
            tag="fp8 row gemm only | fp8_fast_accum=True",
        )
        _run_benchmark(
            row_gemm_bench_no_fast_acc,
            shape=shape,
            tag="fp8 row gemm only | fp8_fast_accum=False",
        )
        _run_benchmark(
            row_gemm_bench_imprecise_acc,
            shape=shape,
            tag="fp8 row gemm only | max_num_imprecise_acc=32",
        )
        _run_benchmark(block_gemm_bench, shape=shape, tag="fp8 block gemm only")
        if rowwise_tma:
            _run_benchmark(
                row_gemm_bench_tma,
                shape=shape,
                tag="fp8 row gemm only | fp8_fast_accum=True | tma_persistent=True",
            )


def bf16_bench(x: Tensor, w: Tensor) -> Callable[[], Tensor]:
    def gemm_fn() -> Tensor:
        return torch.matmul(x, w.T)

    return gemm_fn


def scale_row_bench(x: Tensor, w: Tensor) -> Callable[[], Tensor]:
    # Benchmark quantize(x) + gemm for inference.
    def run_gemm() -> Tensor:
        x_fp8: Tensor
        w_fp8: Tensor
        x_scale: Tensor
        w_scale: Tensor
        x_fp8, x_scale = quantize_fp8_row(x)
        w_fp8, w_scale = quantize_fp8_row(w)
        return matmul_fp8_row(
            x_fp8,
            w_fp8,
            x_scale,
            w_scale,
            dot_out_dtype=torch.float32,
            allow_tf32=True,
            fp8_fast_accum=True,
        )

    return run_gemm


def row_gemm_bench(x: Tensor, w: Tensor) -> Callable[[], Tensor]:
    # Benchmark only row-wise gemm, caching scaling.
    x_fp8: Tensor
    w_fp8: Tensor
    x_scale: Tensor
    w_scale: Tensor
    x_fp8, x_scale = quantize_fp8_row(x)
    w_fp8, w_scale = quantize_fp8_row(w)

    def run_gemm() -> Tensor:
        return matmul_fp8_row(
            x_fp8,
            w_fp8,
            x_scale,
            w_scale,
            dot_out_dtype=torch.float32,
            allow_tf32=True,
            fp8_fast_accum=True,
        )

    return run_gemm


def row_gemm_bench_tma(x: Tensor, w: Tensor) -> Callable[[], Tensor]:
    # Benchmark only row-wise gemm with TMA persistent
    x_fp8: Tensor
    w_fp8: Tensor
    x_scale: Tensor
    w_scale: Tensor
    x_fp8, x_scale = quantize_fp8_row(x)
    w_fp8, w_scale = quantize_fp8_row(w)

    def run_gemm() -> Tensor:
        return matmul_fp8_row(
            x_fp8,
            w_fp8,
            x_scale,
            w_scale,
            dot_out_dtype=torch.float32,
            allow_tf32=True,
            fp8_fast_accum=True,
            tma_persistent=True,
        )

    return run_gemm


def row_gemm_bench_no_fast_acc(x: Tensor, w: Tensor) -> Callable[[], Tensor]:
    # Benchmark only row-wise gemm, caching scaling.
    x_fp8: Tensor
    w_fp8: Tensor
    x_scale: Tensor
    w_scale: Tensor
    x_fp8, x_scale = quantize_fp8_row(x)
    w_fp8, w_scale = quantize_fp8_row(w)

    def run_gemm() -> Tensor:
        return matmul_fp8_row(
            x_fp8,
            w_fp8,
            x_scale,
            w_scale,
            dot_out_dtype=torch.float32,
            allow_tf32=True,
            fp8_fast_accum=False,
        )

    return run_gemm


def row_gemm_bench_imprecise_acc(x: Tensor, w: Tensor) -> Callable[[], Tensor]:
    # Benchmark only row-wise gemm, caching scaling.
    x_fp8: Tensor
    w_fp8: Tensor
    x_scale: Tensor
    w_scale: Tensor
    x_fp8, x_scale = quantize_fp8_row(x)
    w_fp8, w_scale = quantize_fp8_row(w)

    def run_gemm() -> Tensor:
        return matmul_fp8_row(
            x_fp8,
            w_fp8,
            x_scale,
            w_scale,
            dot_out_dtype=torch.float32,
            allow_tf32=True,
            fp8_fast_accum=True,
            imprecise_acc=True,
        )

    return run_gemm


def scale_block_bench(x: Tensor, w: Tensor) -> Callable[[], Tensor]:
    def run_gemm() -> Tensor:
        x_fp8: Tensor
        w_fp8: Tensor
        x_scale: Tensor
        w_scale: Tensor
        x_fp8, x_scale = quantize_fp8_block(x)
        w_fp8, w_scale = quantize_fp8_block(w)
        return matmul_fp8_block(
            x_fp8,
            w_fp8,
            x_scale,
            w_scale,
            dot_out_dtype=torch.float32,
            allow_tf32=True,
            fp8_fast_accum=True,
        )

    return run_gemm


def block_gemm_bench(x: Tensor, w: Tensor) -> Callable[[], Tensor]:
    # Benchmark only block-wise gemm, caching scaling.
    x_fp8: Tensor
    w_fp8: Tensor
    x_scale: Tensor
    w_scale: Tensor
    x_fp8, x_scale = quantize_fp8_block(x)
    w_fp8, w_scale = quantize_fp8_block(w)

    def run_gemm() -> Tensor:
        return matmul_fp8_block(
            x_fp8,
            w_fp8,
            x_scale,
            w_scale,
            dot_out_dtype=torch.float32,
            allow_tf32=True,
            fp8_fast_accum=True,
        )

    return run_gemm


if __name__ == "__main__":
    bench()
