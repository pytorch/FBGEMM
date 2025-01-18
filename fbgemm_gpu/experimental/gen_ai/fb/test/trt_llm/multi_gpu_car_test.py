# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
# pyre-ignore-all-errors[56]

import logging
import os
import tempfile
import unittest
import uuid

import fbgemm_gpu.experimental.fb.gen_ai.trt_llm as trt_llm  # noqa: F401

import torch
from fbgemm_gpu.test.test_utils import gpu_unavailable, running_on_rocm
from hypothesis import given, settings, strategies as st, Verbosity
from torch.distributed.launcher.api import elastic_launch, LaunchConfig

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_cudart_version() -> int:
    versions = [int(v) for v in torch.version.cuda.split(".")]
    versions += [0] * (3 - len(versions))
    return versions[0] * 1000 + versions[1] * 10 + versions[2]


def _run_trt_llm_fused_one_shot_allreduce_residual_rms_norm(
    path: str, dtype: torch.dtype
) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=f"file://{os.path.join(path, 'gloo_rdvz')}",
        world_size=world_size,
        rank=rank,
    )

    hidden_size = 8192
    max_num_seqs = 8192
    eps = 1e-6

    trt_llm.fused_one_shot_allreduce_residual_rms_norm_init(
        device=device,
        rank=rank,
        world_size=world_size,
        max_num_seqs=max_num_seqs,
        hidden_size=hidden_size,
    )

    rtol = 1.3e-6 if dtype == torch.float else 0.1
    atol = 1e-5 if dtype == torch.float else 0.05

    results_ref = []
    results_test = []
    num_seqs = (1, 4, 16, 64, 256, 1024, 2048, max_num_seqs)
    for num_seq in num_seqs:
        x = torch.randn(num_seq, hidden_size, device=device, dtype=dtype)
        residual = torch.randn_like(x)

        y_test = trt_llm.fused_one_shot_allreduce_residual_rms_norm(
            x,
            residual,
            eps,
        )

        rms_norm = torch.nn.RMSNorm(
            hidden_size,
            eps=eps,
            elementwise_affine=False,
            device=device,
            dtype=dtype,
        )
        y_ref = x.clone().detach()
        torch.distributed.all_reduce(y_ref, op=torch.distributed.ReduceOp.SUM)
        y_ref.add_(residual)
        y_ref = rms_norm(y_ref)

        results_ref.append(y_ref)
        results_test.append(y_test)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    for y_ref, y_test, num_seq in zip(results_ref, results_test, num_seqs):
        torch.testing.assert_close(
            y_ref,
            y_test,
            rtol=rtol,
            atol=atol,
            msg="Tensors not close: num_seq {}".format(num_seq),
        )


@unittest.skipIf(*gpu_unavailable)
@unittest.skipIf(*running_on_rocm)
@unittest.skipIf(
    torch.cuda.device_count() < 2, "Not enough GPUs (required at least two GPUs)"
)
@unittest.skipIf(get_cudart_version() < 12040, "CUDA runtime must be >= 12.4")
class TensorRTLLMMultiGpuTests(unittest.TestCase):
    @given(
        dtype=st.sampled_from(
            [
                torch.bfloat16,
                torch.float16,
                torch.float,
            ]
        )
    )
    @settings(verbosity=Verbosity.verbose, max_examples=3, deadline=60000)
    def test_trt_llm_fused_one_shot_allreduce_residual_rms_norm(
        self, dtype: torch.dtype
    ) -> None:
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
            elastic_launch(
                config=lc,
                entrypoint=_run_trt_llm_fused_one_shot_allreduce_residual_rms_norm,
            )(path, dtype)
