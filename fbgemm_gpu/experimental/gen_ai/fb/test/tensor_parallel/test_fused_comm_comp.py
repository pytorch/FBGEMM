# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
# pyre-ignore-all-errors[56]

import logging
import os
import tempfile
import unittest
import uuid
from typing import List

import fbgemm_gpu.experimental.gen_ai  # noqa: F401
import hypothesis.strategies as st

import torch
from hypothesis import given, settings, Verbosity
from torch.distributed.launcher.api import elastic_launch, LaunchConfig

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VERBOSITY: Verbosity = Verbosity.verbose


def _get_tensor(
    shape: List[int],
    dtype: torch.dtype,
    device: torch.device,
    broadcast: bool = True,
) -> torch.Tensor:
    T = torch.rand(*shape, dtype=dtype, device=device) - 0.5
    # Broadcast to ensure all ranks get same values
    if broadcast:
        _ = torch.distributed.broadcast(T, src=0, async_op=False)
    return T


def _check_tensor_all_close(
    rank: int, T: torch.Tensor, T_ref: torch.Tensor, atol=None, rtol=None
) -> None:
    assert not torch.isnan(T).any().item()
    assert not torch.isnan(T_ref).any().item()
    assert not torch.isinf(T).any().item()
    assert not torch.isinf(T_ref).any().item()
    sum_val = torch.sum(T).item()
    sum_val_ref = torch.sum(T_ref).item()
    max_abs_diff = torch.max(torch.abs(T - T_ref)).item()
    if rank == 0:
        print(
            f"Sum val: {sum_val} vs. sum val ref: {sum_val_ref}, max diff: "
            f"{max_abs_diff}"
        )
    torch.testing.assert_close(T, T_ref, rtol=rtol, atol=atol)


def _init_dist_env(win_buff_size: int):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)

    # NCCL envs for the nccl window
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["NCCL_FIRST_COMM_AS_WORLD"] = "true"
    os.environ["NCCL_WIN_NVL_ONLY"] = "true"
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )
    torch.distributed.barrier()

    # Compute the overlapped version
    fused_comm_comp = torch.classes.fbgemm.FusedCommComp(
        rank,  # TP ID
        world_size,  # TP size
        rank,  # TP local ID
        world_size,  # TP local size
        rank,  # Gloabl rank ID
        world_size,  # Global world size
        win_buff_size,  # Number of elements in the buffer
        torch.get_default_dtype(),  # dtype
        0,  # Pointer to the TP communicator, and 0 stands for nullptr
    )
    return (rank, world_size, fused_comm_comp)


def _run_fused_all_gather_gemm(
    M: int, N: int, K: int, transb: bool, inplaceA: bool, barrier: bool
) -> None:
    rank, world_size, fused_comm_comp = _init_dist_env(M * K)
    dtype = torch.get_default_dtype()
    device = torch.cuda.current_device()

    # Generate input matrices, and get the local shard of matrix A
    matA = _get_tensor([M, K], dtype, device)
    matB = _get_tensor([N, K] if transb else [K, N], dtype, device)
    localA = torch.split(matA, (M // world_size), dim=0)[rank]

    # Compute the baseline for references
    ref_local_A = torch.clone(localA).detach()
    ref_ag_A = torch.empty_like(matA)
    _ = torch.distributed.all_gather_into_tensor(ref_ag_A, ref_local_A, async_op=False)

    ref_mat_B = torch.clone(matB).detach()
    ref_mat_B = ref_mat_B.t() if transb else ref_mat_B
    ref_mat_C = torch.matmul(ref_ag_A, ref_mat_B)

    mat_C = torch.empty_like(ref_mat_C)
    dup_A = None
    if inplaceA:
        dup_A = fused_comm_comp.internal_buffer_like(rank * localA.numel(), localA)
        dup_A.copy_(localA, non_blocking=False)

    ag_A, mat_C = fused_comm_comp.split_overlap_ag(
        dup_A if inplaceA else localA,
        False,
        matB,
        transb,
        mat_C,
        inplaceA,
        barrier,
    )

    # Check correctness
    _check_tensor_all_close(rank, ag_A, ref_ag_A)
    _check_tensor_all_close(rank, mat_C, ref_mat_C)


def _run_fused_gemm_reduce_scatter(
    M: int, N: int, K: int, transb: bool, barrier: bool
) -> None:
    rank, world_size, fused_comm_comp = _init_dist_env(M * N)
    dtype = torch.get_default_dtype()
    device = torch.cuda.current_device()

    # Generate input matrices, and get the local shard of matrix A and B
    matA = _get_tensor([M, K], dtype, device)
    matB = _get_tensor([K, N], dtype, device)
    localA = torch.split(matA, (K // world_size), dim=1)[rank]
    localB = torch.split(matB, (K // world_size), dim=0)[rank]
    localA = localA.contiguous()
    if transb:
        localB = localB.t()
    localB = localB.contiguous()

    # Compute the baseline for references
    ref_local_A = torch.clone(localA).detach()
    ref_local_B = torch.clone(localB).detach()
    if transb:
        ref_local_B = ref_local_B.t()
    ref_partial_C = torch.matmul(ref_local_A, ref_local_B)
    ref_ag_C = torch.empty([world_size, M, N], dtype=dtype, device=device)
    _ = torch.distributed.all_gather_into_tensor(
        ref_ag_C, ref_partial_C, async_op=False
    )

    ref_mat_C_shape = list(ref_partial_C.shape)
    ref_mat_C_shape[0] = ref_mat_C_shape[0] // world_size
    ref_mat_C = torch.empty(*ref_mat_C_shape, dtype=dtype, device=device)
    _ = torch.distributed.reduce_scatter_tensor(
        ref_mat_C, ref_partial_C, async_op=False
    )

    mat_C = torch.empty_like(ref_mat_C)
    mat_C = fused_comm_comp.split_overlap_rs(
        localA,
        False,
        localB,
        transb,
        mat_C,
        barrier,
    )

    # Check internal buffer to make sure the final numerical gap results from
    # a reduction order different from ncclReduceScatter
    scatter_C = fused_comm_comp.internal_buffer_like(
        ref_partial_C.numel(), ref_partial_C
    )
    for i in range(world_size):
        src_rank_id = (rank - i + world_size) % world_size
        _check_tensor_all_close(
            rank,
            scatter_C.view(world_size, -1, N)[i],
            ref_ag_C[src_rank_id].view(world_size, -1, N)[rank],
        )

    # Check correctness
    _check_tensor_all_close(rank, mat_C, ref_mat_C, atol=0.6, rtol=0.016)


def _run_rma_win_all_gather(M: int, N: int) -> None:
    rank, world_size, fused_comm_comp = _init_dist_env(M * N)
    dtype = torch.get_default_dtype()
    device = torch.cuda.current_device()

    # Generate input matrices, and get the local shard of matrix A
    ref_mat_A = _get_tensor([M, N], dtype, device)
    localA = torch.split(ref_mat_A, (M // world_size), dim=0)[rank]

    # Because localA is a local shard of matrix A, we expect the all-gather of
    # local shard A will result in a full matrix A
    mat_A = fused_comm_comp.rma_win_all_gather(localA)
    fused_comm_comp.wait_for_comm_stream()

    # Check correctness
    _check_tensor_all_close(rank, mat_A, ref_mat_A)


def _run_rma_win_reduce_scatter(M: int, N: int, flag: bool) -> None:
    rank, world_size, fused_comm_comp = _init_dist_env(M * N)
    dtype = torch.get_default_dtype()
    device = torch.cuda.current_device()

    mat_A = _get_tensor([M, N], dtype, device, broadcast=False)
    ref_local_A = torch.empty([M // world_size, N], dtype=dtype, device=device)
    _ = torch.distributed.reduce_scatter_tensor(ref_local_A, mat_A, async_op=False)
    ref_ag_A = torch.empty([world_size, M, N], dtype=dtype, device=device)
    _ = torch.distributed.all_gather_into_tensor(ref_ag_A, mat_A, async_op=False)

    local_A = torch.empty_like(ref_local_A)
    fused_comm_comp.avoid_incast_congestion(flag)
    fused_comm_comp.ram_win_scatter(mat_A)
    fused_comm_comp.local_reduce_into_tensor(local_A, mat_A)

    # Check internal buffer to make sure the final numerical gap results from
    # a reduction order different from ncclReduceScatter
    scatter_A = fused_comm_comp.internal_buffer_like(mat_A.numel(), mat_A)
    for i in range(world_size):
        src_rank_id = (rank - i + world_size) % world_size
        _check_tensor_all_close(
            rank,
            scatter_A.view(world_size, -1, N)[i],
            ref_ag_A[src_rank_id].view(world_size, -1, N)[rank],
        )

    # Check correctness
    _check_tensor_all_close(rank, local_A, ref_local_A, atol=0.04, rtol=0.016)


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    "These tests require at least 2 GPUs",
)
class FusedCommCompTest(unittest.TestCase):
    @settings(verbosity=VERBOSITY, max_examples=1, deadline=None)
    @given(
        seqlen=st.sampled_from([8192]),
        model_dim=st.sampled_from([16384]),
        ffn_dim=st.sampled_from([16384]),
        transb=st.sampled_from([False]),
        inplaceA=st.sampled_from([True]),
        barrier=st.sampled_from([False]),
    )
    def test_fused_all_gather_gemm(
        self,
        seqlen: int,
        model_dim: int,
        ffn_dim: int,
        transb: bool,
        inplaceA: bool,
        barrier: bool,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
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
            elastic_launch(config=lc, entrypoint=_run_fused_all_gather_gemm)(
                seqlen,  # M
                ffn_dim,  # N
                model_dim,  # K
                transb,
                inplaceA,
                barrier,
            )

    @settings(verbosity=VERBOSITY, max_examples=1, deadline=None)
    @given(
        seqlen=st.sampled_from([8192]),
        model_dim=st.sampled_from([16384]),
        ffn_dim=st.sampled_from([16384]),
        transb=st.sampled_from([False]),
        barrier=st.sampled_from([False]),
    )
    def test_fused_gemm_reduce_scatter(
        self,
        seqlen: int,
        model_dim: int,
        ffn_dim: int,
        transb: bool,
        barrier: bool,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
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
            elastic_launch(config=lc, entrypoint=_run_fused_gemm_reduce_scatter)(  # noqa: E501
                seqlen,  # M
                model_dim,  # N
                ffn_dim,  # K
                transb,
                barrier,
            )

    @settings(verbosity=VERBOSITY, max_examples=1, deadline=None)
    @given(
        seqlen=st.sampled_from([8192]),
        model_dim=st.sampled_from([16384]),
    )
    def test_rma_win_all_gather(
        self,
        seqlen: int,
        model_dim: int,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
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
            elastic_launch(config=lc, entrypoint=_run_rma_win_all_gather)(
                seqlen, model_dim
            )

    @settings(verbosity=VERBOSITY, max_examples=2, deadline=None)
    @given(
        seqlen=st.sampled_from([8192]),
        model_dim=st.sampled_from([16384]),
        avoid_incast_congestion=st.sampled_from([False, True]),
    )
    def test_rma_win_reduce_scatter(
        self,
        seqlen: int,
        model_dim: int,
        avoid_incast_congestion: bool,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
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
            elastic_launch(config=lc, entrypoint=_run_rma_win_reduce_scatter)(
                seqlen, model_dim, avoid_incast_congestion
            )
