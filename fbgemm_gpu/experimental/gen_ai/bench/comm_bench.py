# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse

import os
import tempfile
import uuid
from functools import lru_cache
from pprint import pprint
from typing import Tuple

import fbgemm_gpu.experimental.gen_ai  # noqa: F401
import pandas as pd

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.launcher.api import elastic_launch, LaunchConfig


@lru_cache(None)
def get_symm_buffer(group):
    inp = symm_mem.empty(
        16 * 1024 * 1024, device="cuda", dtype=torch.bfloat16
    )  # .normal_()
    symm_mem.rendezvous(inp, group=group)
    return inp, group.group_name


def _setup(path: str) -> Tuple[int, int]:
    rank = int(os.environ["LOCAL_RANK"])
    W = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

    torch.ops.fbgemm.nccl_init(rank, W, os.path.join(path, "rdvz"))
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=f"file://{os.path.join(path, 'gloo_rdvz')}",
        world_size=W,
        rank=rank,
    )

    buffer = torch.ops.fbgemm.car_tensor()
    barrier = torch.ops.fbgemm.car_tensor()
    barrier.zero_()

    buffer_handle = torch.ops.fbgemm.car_ipc_handle(buffer)
    all_buffer_handles = [torch.empty_like(buffer_handle) for _ in range(W)]
    torch.distributed.all_gather(all_buffer_handles, buffer_handle)

    barrier_handle = torch.ops.fbgemm.car_ipc_handle(barrier)
    all_barrier_handles = [torch.empty_like(barrier_handle) for _ in range(W)]
    torch.distributed.all_gather(all_barrier_handles, barrier_handle)
    torch.ops.fbgemm.car_init(
        rank, W, barrier, all_barrier_handles, buffer, all_buffer_handles
    )
    torch.cuda.synchronize()
    torch.distributed.barrier()
    group = dist.group.WORLD
    _ = get_symm_buffer(group)
    return rank, W


def symm_one_shot_allreduce(dst_tensor, src_tensor, bias=None, comm_idx=None):
    # get_symm_buffer should be called for the first time during model init,
    # and now return cached values. Make sure group is the same as during init
    symm_buffer, group_name = get_symm_buffer(dist.group.WORLD)
    symm_buffer = symm_buffer[: src_tensor.numel()].view_as(src_tensor)
    torch.ops.symm_mem.one_shot_all_reduce_copy_out(
        symm_buffer, src_tensor, "sum", group_name, dst_tensor
    )
    if bias is not None:
        dst_tensor.add_(bias)


def symm_two_shot_allreduce(dst_tensor, src_tensor, bias=None, comm_idx=None):
    # get_symm_buffer should be called for the first time during model init,
    # and now return cached values. Make sure group is the same as during init
    symm_buffer, group_name = get_symm_buffer(dist.group.WORLD)
    # car is also doing explicit copy
    symm_buffer = symm_buffer[: src_tensor.numel()].view_as(src_tensor)
    symm_buffer.copy_(src_tensor)
    torch.ops.symm_mem.two_shot_all_reduce_out(
        symm_buffer, "sum", group_name, dst_tensor
    )
    if bias is not None:
        dst_tensor.add_(bias)


def symm_reduce_scatter(dst_tensor, src_tensor, comm_idx=None):
    symm_buffer, group_name = get_symm_buffer(dist.group.WORLD)
    symm_buffer = symm_buffer[: src_tensor.numel()].view_as(src_tensor)
    symm_buffer.copy_(src_tensor)
    torch.ops.symm_mem.reduce_scatter_out(symm_buffer, group_name, False, dst_tensor)


def run_one_algo(fn, out, inp, num_iters, num_warmup_iters):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(num_warmup_iters):
        fn(out, inp)
    start_event.record()
    for _ in range(num_iters):
        fn(out, inp)
    end_event.record()
    torch.cuda.synchronize()
    time = start_event.elapsed_time(end_event) / num_iters
    return time


def run_benchmark(args, path):
    rank, W = _setup(path)
    if rank == 0:
        print(f"Running benchmark with {W} ranks")
    # benchmark_results = defaultdict(defaultdict)
    benchmark_results = []
    # with torch.profiler.profile() as p:
    for N in torch.logspace(
        args.min_size, args.max_size, steps=args.size_steps, base=2
    ).tolist():

        def round_up(a: int, b: int) -> int:
            return ((a + b - 1) // b) * b

        N_even_divisor = 8 * 64 if torch.version.hip else 8 * 32
        N = round_up(int(N), N_even_divisor)
        inp = torch.rand(N, dtype=torch.bfloat16, device="cuda")
        results = {"N": N}
        if args.op == "allreduce":
            out = torch.full_like(inp, -1)
            fns = (
                torch.ops.fbgemm.one_shot_car_allreduce,
                symm_one_shot_allreduce,
                torch.ops.fbgemm.two_shot_car_allreduce,
                symm_two_shot_allreduce,
                torch.ops.fbgemm.nccl_allreduce,
            )
            labels = (
                "fbgemm_1shot",
                "symm_1shot",
                "fbgemm_2shot",
                "symm_2shot",
                "nccl",
            )
            for fn, label in zip(fns, labels):
                time = run_one_algo(
                    fn,
                    out,
                    inp,
                    args.num_iters,
                    args.num_warmup_iters,
                )
                results[f"{label}_time"] = time
                results[f"{label}_bwidth"] = (
                    N * inp.element_size() / (time * 1e-3) / 1e9
                )
        else:
            out = torch.full(
                (inp.shape[0] // W,), -1, dtype=inp.dtype, device=inp.device
            )
            fns = (
                torch.ops.fbgemm.car_reducescatter,
                symm_reduce_scatter,
                torch.ops.fbgemm.nccl_reducescatter,
            )
            labels = ("fbgemm_rs", "symm_rs", "nccl_rs")
            for fn, label in zip(fns, labels):
                time = run_one_algo(
                    fn,
                    out,
                    inp,
                    args.num_iters,
                    args.num_warmup_iters,
                )
                results[f"{label}_time"] = time
                results[f"{label}_bwidth"] = (
                    N * inp.element_size() / (time * 1e-3) / 1e9
                )

        benchmark_results.append(results)

    if rank == 0:
        pprint(benchmark_results)
        if args.export_csv:
            csv_file = os.path.join(args.output_dir, "comm_ops_benchmark.csv")
            # Export results to a CSV file.
            df = pd.DataFrame(benchmark_results)
            df.to_csv(csv_file, index=False)


def main(args, path):
    if args.export_csv:
        os.makedirs(args.output_dir, exist_ok=True)
        print("csv and images will be saved to " + args.output_dir)

    lc = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=args.num_ranks,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        monitor_interval=1,
    )
    elastic_launch(lc, entrypoint=run_benchmark)(args, path)


def invoke_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="/tmp", help="Directory to save plots and csvs to"
    )
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Export results to a CSV file.",
    )
    parser.add_argument("--num_ranks", type=int, default=8)
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--num_warmup_iters", type=int, default=10)
    parser.add_argument(
        "--min_size",
        type=int,
        default=10,
        help="minimum size will be set to 2**min_size",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=24,
        help="maximum size will be set to 2**max_size",
    )
    parser.add_argument(
        "--size_steps", type=int, default=20, help="number of size steps to run"
    )
    parser.add_argument(
        "--op",
        type=str,
        default="allreduce",
        choices=["allreduce", "reduce_scatter"],
        help="op to benchmark, allreduce or reduce_scatter",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as path:
        main(args, path)


if __name__ == "__main__":
    invoke_main()
