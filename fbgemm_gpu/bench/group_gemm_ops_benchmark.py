#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
from typing import List

import click
import fbgemm_gpu
import torch

from torch.profiler import profile, ProfilerActivity

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from bench_utils import benchmark_torch_function
else:
    from fbgemm_gpu.bench.bench_utils import benchmark_torch_function

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:group_gemm_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:group_gemm_ops_cpu")


torch.backends.cuda.matmul.allow_tf32 = True


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--num-groups", default=8, type=int)
@click.option("--a-num-rows", default=1024, type=int)
@click.option("--a-num-cols", default=1024, type=int)
@click.option("--b-num-cols", default=1024, type=int)
@click.option("--data-type", default="float", type=str)
@click.option("--beta", default=0, type=int)
@click.option("--c-num-dims", default=1, type=int)
def fixed_shapes_bench(
    num_groups: int,
    a_num_rows: int,
    a_num_cols: int,
    b_num_cols: int,
    data_type: str,
    beta: int,  # Can be 0 or 1. If beta=1, C will be added to A * B
    c_num_dims: int,  # C tensor can be 1D or 2D
) -> None:
    """
    Benchmark A * B + beta * C
    """
    assert a_num_cols % 8 == 0
    assert b_num_cols % 8 == 0
    assert beta == 0 or beta == 1
    assert c_num_dims == 1 or c_num_dims == 2

    if data_type == "half":
        dtype = torch.half
    elif data_type == "float":
        dtype = torch.float
    elif data_type == "double":
        dtype = torch.double
    else:
        raise ValueError(f"Data type {data_type} is not supported")

    a_group = [
        torch.rand((a_num_rows, a_num_cols), dtype=dtype, device="cuda")
        for _ in range(num_groups)
    ]
    b_group = [
        torch.rand((a_num_cols, b_num_cols), dtype=dtype, device="cuda")
        for _ in range(num_groups)
    ]
    if beta == 1:
        c_shape = (b_num_cols,) if c_num_dims == 1 else (a_num_rows, b_num_cols)
        c_group = [
            torch.rand(c_shape, dtype=dtype, device="cuda") for _ in range(num_groups)
        ]
    else:
        c_group = None

    t, _ = benchmark_torch_function(
        torch.ops.fbgemm.gmm,
        (a_group, b_group, c_group),
        flush_gpu_cache_size_mb=0,
    )
    flops = num_groups * a_num_rows * a_num_cols * b_num_cols * 2 + (
        num_groups * a_num_rows * b_num_cols if beta == 1 else 0
    )
    print(f"{t} sec {flops / t / 1e12} TF/s")

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        benchmark_torch_function(
            torch.ops.fbgemm.gmm,
            (
                a_group,
                b_group,
                c_group,
            ),
            flush_gpu_cache_size_mb=0,
        )
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("gmm_trace.json")


def gmm_ref(
    a_group: List[torch.Tensor], b_group: List[torch.Tensor]
) -> List[torch.Tensor]:
    output = []
    for a, b in zip(a_group, b_group):
        output.append(torch.mm(a, b))
    return output


def gmm_add_ref(
    a_group: List[torch.Tensor],
    b_group: List[torch.Tensor],
    c_group: List[torch.Tensor],
) -> List[torch.Tensor]:
    output = []
    for a, b, c in zip(a_group, b_group, c_group):
        output.append(torch.addmm(c, a, b))
    return output


@cli.command()
@click.option("--data-type", default="float")
@click.option(
    "--sizes",
    type=str,
    default="['2314x96x192',"
    "'235520x96x192',"
    "'1909x96x192',"
    "'235520x96x192',"
    "'235520x96x192',"
    "'61440x96x192',"
    "'10240x96x192',"
    "'70847x96x192',"
    "'68863x96x192',"
    "'2265x96x192',"
    "'512000x96x192',"
    "'122880x96x192',"
    "'20480x96x192']",
)
@click.option("--pad-n", is_flag=True)
@click.option("--beta", default=0, type=int)
@click.option("--c-num-dims", default=1, type=int)
@click.option("--enable-trace-profile", is_flag=True)
def custom_shapes_bench(
    data_type: str,
    sizes: str,
    pad_n: bool,
    beta: int,  # Can be 0 or 1. If beta=1, C will be added to A * B
    c_num_dims: int,  # C tensor can be 1D or 2D
    enable_trace_profile: bool,
) -> None:
    assert beta == 0 or beta == 1
    assert c_num_dims == 1 or c_num_dims == 2

    ms = []
    ns = []
    ks = []

    print("{: <8} {: <8} {: <8}".format("m", "n", "k"))
    for mnk in ast.literal_eval(sizes):
        m, n, k = mnk.split("x")
        ms.append(int(m))
        ns.append(int(n))
        ks.append(int(k))
        print(f"{m: <8} {n: <8} {k: <8}")

    if data_type == "half":
        dtype = torch.half
    elif data_type == "float":
        dtype = torch.float
    elif data_type == "double":
        dtype = torch.double
    else:
        raise ValueError(f"Data type {data_type} is not supported")

    print(f"pad_n {pad_n}")

    for i, (n, k) in enumerate(zip(ns, ks)):
        if pad_n and n % 8 != 0:
            n = ((n // 8) + 1) * 8
            ns[i] = n
        else:
            assert n % 8 == 0, f"n % 8 != 0 ({n})"
        assert k % 8 == 0, f"k % 8 != 0 ({k})"

    a_group = []
    b_group = []
    c_group = []
    flops = 0
    for m, n, k in zip(ms, ns, ks):
        a_group.append(torch.rand(m, k, dtype=dtype, device="cuda"))
        b_group.append(torch.rand(k, n, dtype=dtype, device="cuda"))
        if beta == 1:
            c_shape = (n,) if c_num_dims == 1 else (m, n)
            c_group.append(torch.rand(*c_shape, dtype=dtype, device="cuda"))
        flops += m * n * k * 2 + ((m * n) if beta == 1 else 0)

    time, output = benchmark_torch_function(
        torch.ops.fbgemm.gmm,
        (
            a_group,
            b_group,
            c_group if beta == 1 else None,
        ),
        num_warmups=10,
        iters=100,
        flush_gpu_cache_size_mb=0,
    )

    time_ref, output_ref = benchmark_torch_function(
        gmm_add_ref if beta == 1 else gmm_ref,
        (a_group, b_group, c_group) if beta == 1 else (a_group, b_group),
        num_warmups=10,
        iters=100,
        flush_gpu_cache_size_mb=0,
    )

    print(
        f"custom_shapes_bench: reference ({data_type}) {time_ref} sec {flops / time_ref / 1e12} TF/s"
    )
    print(
        f"custom_shapes_bench: gmm ({data_type}) {time} sec {flops / time / 1e12} TF/s"
    )

    for ref, test in zip(output_ref, output):
        assert torch.allclose(ref, test, rtol=1e-3, atol=1e-3)

    if enable_trace_profile:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _, output = benchmark_torch_function(
                torch.ops.fbgemm.gmm,
                (
                    a_group,
                    b_group,
                ),
                num_warmups=10,
                iters=100,
                flush_gpu_cache_size_mb=0,
            )
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace("gmm_bench.json")


if __name__ == "__main__":
    cli()
