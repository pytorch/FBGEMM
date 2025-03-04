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
from typing import Callable, Dict, List, Tuple, Union

import fbgemm_gpu.experimental.gen_ai  # noqa: F401

import numpy as np
import torch
from hypothesis import given, settings, strategies as st, Verbosity
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

    return rank, W


def _run_allgather_inner(
    rdvz: str,
    dst_dtype: torch.dtype,
    src_dtype: torch.dtype,
    skip_torch_compile: bool = False,
) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    W = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    torch.ops.fbgemm.nccl_init(rank, W, rdvz)

    B, T, D = 2, 4096, 1024
    y = torch.empty(size=(B, T, D), dtype=src_dtype, device="cuda")
    y[:] = rank
    y_gather = torch.full(
        size=(W, B, T, D), fill_value=-1, dtype=dst_dtype, device="cuda"
    )
    # TORCH_CHECK failures can be suppressed by torch.compile, in which case
    # we may not be able to capture the right exception in Python.
    if not skip_torch_compile:
        # Here we test to confirm that allgather is compatible with torch.compile.
        torch.compile(torch.ops.fbgemm.nccl_allgather)(y_gather, y)
        for w in range(W):
            torch.testing.assert_close(
                y_gather[w],
                torch.full(
                    size=(B, T, D), fill_value=w, dtype=dst_dtype, device=y.device
                ),
            )

    for _ in range(20):
        torch.ops.fbgemm.nccl_allgather(y_gather, y)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.ops.fbgemm.nccl_allgather(y_gather, y)

    for _ in range(10):
        g.replay()


def _run_reducescatter_inner(path: str) -> None:
    rank, W = _setup(path)

    # Test to make sure reducescatter is compatible with torch.compile.
    reducescatter_compiled = torch.compile(torch.ops.fbgemm.nccl_reducescatter)
    car_reducescatter_compiled = torch.compile(torch.ops.fbgemm.car_reducescatter)

    def round_up(a: int, b: int) -> int:
        return int(math.ceil(a / b)) * b

    def _test_fn(
        fn: Callable,  # pyre-ignore
        W: int,
        rank: int,
        roundup: int,
        split_last_dim: bool = False,
    ) -> None:
        for N in np.logspace(10, 24, num=20, base=2).tolist():
            N = round_up(int(N), roundup)
            y = torch.zeros(size=(N,), dtype=torch.bfloat16, device="cuda")
            y[:] = rank
            y_reducescatter = torch.empty(
                y.numel() // W, dtype=y.dtype, device=y.device
            )
            rank_start = N // W * rank
            rank_end = N // W * (rank + 1)
            args: List[torch.Tensor] = [y_reducescatter, y]
            kwargs: Dict[str, Union[bool, torch.Tensor]] = {}

            if split_last_dim:
                kwargs["split_last_dim"] = True

            fn(*args, **kwargs)
            target = torch.full(
                size=(N,),
                fill_value=(W * (W - 1) // 2),
                dtype=torch.bfloat16,
                device=y.device,
            )

            torch.testing.assert_close(
                y_reducescatter,
                target[rank_start:rank_end],
            )

    # nccl allreduce doesn't support split_last_dim
    _test_fn(reducescatter_compiled, W, rank, W)
    _test_fn(reducescatter_compiled, W, rank, W)

    _test_fn(car_reducescatter_compiled, W, rank, 1024, False)
    _test_fn(car_reducescatter_compiled, W, rank, 1024, True)


def _run_allreduce_inner(path: str) -> None:
    rank, W = _setup(path)

    # Test to make sure allreduce is compatible with torch.compile.
    allreduce_compiled = torch.compile(torch.ops.fbgemm.nccl_allreduce)

    for N in np.logspace(10, 24, num=20, base=2).tolist():
        N = int(N)
        y = torch.zeros(size=(N,), dtype=torch.bfloat16, device="cuda")
        y[:] = rank
        y_allreduce = torch.empty_like(y)
        allreduce_compiled(y_allreduce, y)
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
        allreduce_compiled(y_allreduce, y, z)
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

        N_even_divisor = 8 * 64 if torch.version.hip else 8 * 32
        N = round_up(N, N_even_divisor)
        if rank == 0:
            logger.info(f"N: {N}")
        y = torch.zeros(size=(N,), dtype=torch.bfloat16, device="cuda")
        y[:] = rank
        y_allreduce = torch.empty_like(y)
        one_shot_allreduce_compiled = torch.compile(
            torch.ops.fbgemm.one_shot_car_allreduce
        )
        one_shot_allreduce_compiled(y_allreduce, y)
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
        one_shot_allreduce_compiled(y_allreduce, y, z)
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
            two_shot_allreduce_compiled = torch.compile(
                torch.ops.fbgemm.two_shot_car_allreduce
            )
            two_shot_allreduce_compiled(y_allreduce, y)
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
            two_shot_allreduce_compiled(y_allreduce, y, z)
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
    rank, W = _setup(path)

    ITER = 1000
    for idx, N in enumerate([0] + np.logspace(4, 24, num=20, base=2).tolist()):
        N = int(N)

        def round_up(a: int, b: int) -> int:
            return int(math.ceil(a / b)) * b

        N_even_divisor = 8 * 64 if torch.version.hip else 8 * 32
        N = round_up(N, N_even_divisor)
        if rank == 0:
            logger.info(f"N: {N}")
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
    @given(
        dtype=st.sampled_from(
            [
                torch.bfloat16,
                torch.float16,
                torch.int,
                torch.long,
                torch.float,
                torch.float8_e4m3fn,
            ]
        )
    )
    @settings(verbosity=Verbosity.verbose, max_examples=3, deadline=100000)
    def test_allgather(self, dtype: torch.dtype) -> None:
        # float8 is only supported in H100 or MI300x
        if dtype == torch.float8_e4m3fn:
            if torch.version.hip:
                dtype = torch.float8_e4m3fnuz
            elif torch.cuda.get_device_capability() < (9, 0):
                self.skipTest(
                    "float8_e4m3fn is only supported in H100 or MI300x, but we're running "
                    f"on {torch.cuda.get_device_capability()}"
                )

        with tempfile.TemporaryDirectory() as path:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=torch.cuda.device_count(),
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint="localhost:0",
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
                local_addr="localhost",
            )
            elastic_launch(config=lc, entrypoint=_run_allgather_inner)(
                os.path.join(path, "rdvz"), dtype, dtype
            )

    @given(
        dtypes=st.sampled_from(
            [
                (torch.bfloat16, torch.float16),
                (torch.bfloat16, torch.int),
                (torch.bfloat16, torch.float8_e4m3fn),
            ]
        )
    )
    @settings(verbosity=Verbosity.verbose, max_examples=3, deadline=100000)
    def test_allgather_dtype_mismatch(
        self, dtypes: Tuple[torch.dtype, torch.dtype]
    ) -> None:
        dst_dtype, src_dtype = dtypes
        # float8 is only supported in H100 or MI300x
        if dst_dtype == torch.float8_e4m3fn or src_dtype == torch.float8_e4m3fn:
            if torch.version.hip:
                if dst_dtype == torch.float8_e4m3fn:
                    dst_dtype = torch.float8_e4m3fnuz
                if src_dtype == torch.float8_e4m3fn:
                    src_dtype = torch.float8_e4m3fnuz
            elif torch.cuda.get_device_capability() < (9, 0):
                self.skipTest(
                    "float8_e4m3fn is only supported in H100 or MI300x, but we're running "
                    f"on {torch.cuda.get_device_capability()}"
                )

        with tempfile.TemporaryDirectory() as path:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=torch.cuda.device_count(),
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint="localhost:0",
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
                local_addr="localhost",
            )
            with self.assertRaises(Exception) as cm:
                elastic_launch(config=lc, entrypoint=_run_allgather_inner)(
                    os.path.join(path, "rdvz"),
                    dst_dtype,
                    src_dtype,
                    True,
                )
            self.assertTrue(
                "dst and src tensors must have the same dtype." in cm.exception.args[0]
            )

    def test_reducescatter(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=torch.cuda.device_count(),
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint="localhost:0",
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
                local_addr="localhost",
            )
            elastic_launch(config=lc, entrypoint=_run_reducescatter_inner)(path)

    def test_allreduce(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=torch.cuda.device_count(),
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint="localhost:0",
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
                local_addr="localhost",
            )
            elastic_launch(config=lc, entrypoint=_run_allreduce_inner)(path)

    def test_oneshot_car_stress(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=torch.cuda.device_count(),
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint="localhost:0",
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
                local_addr="localhost",
            )
            elastic_launch(config=lc, entrypoint=_run_oneshot_car_stress_inner)(path)
