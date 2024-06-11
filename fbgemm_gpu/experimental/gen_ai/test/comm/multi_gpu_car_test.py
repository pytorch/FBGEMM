# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import functools
import logging
import math
import os
import tempfile
import unittest
import uuid

import fbgemm_gpu.experimental.gen_ai  # noqa: F401

import numpy as np
import torch
from torch.distributed.launcher.api import elastic_launch, LaunchConfig

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)


@functools.lru_cache
def has_nvswitch() -> bool:
    import subprocess

    model = subprocess.check_output(
        "cat /etc/fbwhoami | grep MODEL_NAME", shell=True
    ).decode("utf-8")
    return "GRANDTETON" in model or "SUPERMICRO" in model


def _run_allgather_inner(rdvz: str) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    W = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    torch.ops.fbgemm.nccl_init(rank, W, rdvz)
    # torch.distributed.init_process_group(backend="nccl")

    B, T, D = 2, 4096, 1024
    y = torch.randn(size=(B, T, D), dtype=torch.bfloat16, device="cuda")
    y[:] = rank
    y_gather = torch.zeros(size=(W, B, T, D), dtype=torch.bfloat16, device="cuda")
    y_gather[:] = -1
    torch.ops.fbgemm.nccl_allgather(y_gather, y)
    for w in range(W):
        torch.testing.assert_close(
            y_gather[w],
            torch.full(
                size=(B, T, D), fill_value=w, dtype=torch.bfloat16, device=y.device
            ),
        )

    for _ in range(20):
        torch.ops.fbgemm.nccl_allgather(y_gather, y)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.ops.fbgemm.nccl_allgather(y_gather, y)

    for _ in range(10):
        g.replay()


def _run_allreduce_inner(path: str) -> None:
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

    for N in np.logspace(10, 24, num=20, base=2).tolist():
        N = int(N)
        y = torch.zeros(size=(N,), dtype=torch.bfloat16, device="cuda")
        y[:] = rank
        y_allreduce = torch.empty_like(y)
        torch.ops.fbgemm.nccl_allreduce(y_allreduce, y)
        torch.testing.assert_close(
            y_allreduce,
            torch.full(
                size=(N,),
                fill_value=(W * (W - 1) // 2),
                dtype=torch.bfloat16,
                device=y.device,
            ),
        )

        z = torch.ones(size=(N,), dtype=torch.bfloat16, device="cuda")
        torch.ops.fbgemm.nccl_allreduce(y_allreduce, y, z)
        torch.testing.assert_close(
            y_allreduce,
            torch.full(
                size=(N,),
                fill_value=(W * (W - 1) // 2),
                dtype=torch.bfloat16,
                device=y.device,
            )
            + 1,
        )

        def round_up(a: int, b: int) -> int:
            return int(math.ceil(a / b)) * b

        N = round_up(N, 256)
        y = torch.zeros(size=(N,), dtype=torch.bfloat16, device="cuda")
        y[:] = rank
        y_allreduce = torch.empty_like(y)
        torch.ops.fbgemm.one_shot_car_allreduce(y_allreduce, y)
        torch.testing.assert_close(
            y_allreduce,
            torch.full(
                size=(N,),
                fill_value=(W * (W - 1) // 2),
                dtype=torch.bfloat16,
                device=y.device,
            ),
        )
        z = torch.ones(size=(N,), dtype=torch.bfloat16, device="cuda")
        torch.ops.fbgemm.one_shot_car_allreduce(y_allreduce, y, z)
        torch.testing.assert_close(
            y_allreduce,
            torch.full(
                size=(N,),
                fill_value=(W * (W - 1) // 2),
                dtype=torch.bfloat16,
                device=y.device,
            )
            + 1,
        )
        if has_nvswitch() or (not has_nvswitch() and N < 16 * 1024):
            N = round_up(N, 1024)
            y = torch.zeros(size=(N,), dtype=torch.bfloat16, device="cuda")
            y[:] = rank
            y_allreduce = torch.empty_like(y)
            torch.ops.fbgemm.two_shot_car_allreduce(y_allreduce, y)
            torch.testing.assert_close(
                y_allreduce,
                torch.full(
                    size=(N,),
                    fill_value=(W * (W - 1) // 2),
                    dtype=torch.bfloat16,
                    device=y.device,
                ),
            )
            z = torch.ones(size=(N,), dtype=torch.bfloat16, device="cuda")
            torch.ops.fbgemm.two_shot_car_allreduce(y_allreduce, y, z)
            torch.testing.assert_close(
                y_allreduce,
                torch.full(
                    size=(N,),
                    fill_value=(W * (W - 1) // 2),
                    dtype=torch.bfloat16,
                    device=y.device,
                )
                + 1,
            )


def _run_oneshot_car_stress_inner(path: str) -> None:
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

    ITER = 1000
    for idx, N in enumerate(np.logspace(4, 24, num=20, base=2).tolist()):
        N = int(N)

        def round_up(a: int, b: int) -> int:
            return int(math.ceil(a / b)) * b

        N = round_up(N, 256)
        if rank == 0:
            print(f"N: {N}")
        for iterId in range(ITER):
            y = torch.zeros(size=(N,), dtype=torch.bfloat16, device="cuda")
            y[:] = rank + idx + iterId
            y_allreduce = torch.empty_like(y)
            torch.ops.fbgemm.one_shot_car_allreduce(y_allreduce, y)
            torch.testing.assert_close(
                y_allreduce,
                torch.full(
                    size=(N,),
                    fill_value=(W * (W - 1) // 2),
                    dtype=torch.bfloat16,
                    device=y.device,
                )
                + (idx + iterId) * W,
            )


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    "Skip when CUDA is not available or when there are not enough GPUs; these tests require at least two GPUs",
)
class LLamaMultiGpuTests(unittest.TestCase):
    def test_allgather(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as path:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=torch.cuda.device_count(),
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )
            elastic_launch(config=lc, entrypoint=_run_allgather_inner)(
                os.path.join(path, "rdvz")
            )

    def test_allreduce(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as path:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=torch.cuda.device_count(),
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )
            elastic_launch(config=lc, entrypoint=_run_allreduce_inner)(path)

    def test_oneshot_car_stress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as path:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=torch.cuda.device_count(),
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )
            elastic_launch(config=lc, entrypoint=_run_oneshot_car_stress_inner)(path)
