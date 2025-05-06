# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import argparse
import logging
import os
import tempfile
import traceback
from datetime import datetime
from functools import partial

import pytest
import torch
from deeplearning.fbgemm.fbgemm_gpu.experimental.gen_ai.test.moe.parallelism import (
    get_ep_group,
    get_global_rank,
    get_routed_experts_mp_group,
    init_parallel,
)
from fbgemm_gpu.experimental.gen_ai.moe.layers import (
    BaselineMoE,
    MetaShufflingMoE,
    MoEArgs,
)
from torch.distributed import launcher
from torch.distributed.launcher.api import LaunchConfig

# pyre-fixme[21]: Could not find name `ProfilerActivity` in `torch.profiler`.
from torch.profiler import profile, ProfilerActivity

TRACE_DIR: str = "/tmp/"
WARM_UP_ITERS = 15
PROFILE_ITERS = 20


def kineto_trace_handler(
    prof: torch.profiler.profile,
    trace_filename: str,
) -> None:
    os.makedirs(TRACE_DIR, exist_ok=True)
    TRACE_PATH = f"{TRACE_DIR}/{trace_filename}"
    prof.export_chrome_trace(TRACE_PATH)
    logging.info(f"trace saved to {TRACE_PATH} ")


def get_launch_config() -> LaunchConfig:
    return LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=8,
        rdzv_backend=("c10d"),
        rdzv_endpoint="localhost:0",
        rdzv_configs={
            "is_host": True,
            "rank": 0,
        },
        run_id="DEFAULT_RUN_ID",
        max_restarts=1,
        start_method="spawn",
    )


@pytest.mark.node_only
@pytest.mark.parametrize("dim", [4096])
@pytest.mark.parametrize("hidden_dim", [16384])
@pytest.mark.parametrize("ffn_dim_multiplier", [1.2])
@pytest.mark.parametrize("multiple_of", [2048])
@pytest.mark.parametrize("mp_size", [4])
@pytest.mark.parametrize("ep_size", [2])
@pytest.mark.parametrize("num_experts", [128])
@pytest.mark.parametrize("mp_size_for_routed_experts", [4])
@pytest.mark.parametrize("num_local_experts", [64])
@pytest.mark.parametrize("top_k", [1])
@pytest.mark.parametrize("auto_scale_F", [True, False])
@pytest.mark.parametrize("use_fast_accum", [True, False])
@pytest.mark.parametrize("dedup_comm", [True, False])
def test_demo(
    dim: int,
    hidden_dim: int,
    ffn_dim_multiplier: float,
    multiple_of: int,
    mp_size: int,
    ep_size: int,
    num_experts: int,
    mp_size_for_routed_experts: int,
    num_local_experts: int,
    top_k: int,
    auto_scale_F: bool,
    use_fast_accum: bool,
    dedup_comm: bool,
):
    torch.manual_seed(43)
    cmd_args = argparse.Namespace(**locals())
    moe_args = MoEArgs(**vars(cmd_args))
    with tempfile.TemporaryDirectory():
        launcher.elastic_launch(get_launch_config(), entrypoint=run_demo)(moe_args)


def run_demo(args: MoEArgs) -> None:
    try:
        init_parallel(args.mp_size, args.ep_size, args.mp_size_for_routed_experts)

        global_rank = get_global_rank()
        logging.info(f"Running demo in child process at {global_rank=}.")

        ep_group = get_ep_group()
        mp_group = get_routed_experts_mp_group()

        torch.set_default_dtype(torch.bfloat16)
        torch.random.manual_seed(global_rank)
        baseline_moe = BaselineMoE(
            ep_group=ep_group,
            ep_mp_group=mp_group,
            moe_args=args,
        ).build()
        baseline_moe.to("cuda")

        torch.random.manual_seed(global_rank)
        shuffling_moe = MetaShufflingMoE(
            ep_group=ep_group,
            ep_mp_group=mp_group,
            moe_args=args,
        ).build()
        shuffling_moe.to("cuda")

        tokens = torch.empty(
            size=(32, 32, args.dim), device="cuda", dtype=torch.bfloat16
        )
        torch.nn.init.trunc_normal_(tokens, std=0.1, a=-0.2, b=0.2)

        # SharedExperts MP is the same as RoutedExperts MP.
        torch.distributed.broadcast(
            tokens,
            src=torch.distributed.get_global_rank(mp_group, 0),
            group=mp_group,
        )

        baseline_output = baseline_moe(tokens, use_static_shape=True)
        torch.distributed.all_reduce(
            baseline_output,
            group=mp_group,
        )

        shuffling_output = shuffling_moe(tokens, use_static_shape=True)
        torch.distributed.all_reduce(
            shuffling_output,
            group=mp_group,
        )

        logging.info(
            f"{global_rank=}, {baseline_output.norm()=}, {shuffling_output.norm()=}"
        )
        torch.testing.assert_close(
            baseline_output, shuffling_output, atol=1.1e-3, rtol=1.6e-2
        )

        def run_profiling(moe: torch.nn.Module, name: str):
            timestamp = datetime.timestamp(datetime.now())
            trace_filename = f"bench_{name}_{timestamp}_{global_rank}.json"
            for _ in range(WARM_UP_ITERS):
                _ = moe.forward(tokens, use_static_shape=True)
            with profile(
                # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=partial(
                    kineto_trace_handler,
                    trace_filename=trace_filename,
                ),
                record_shapes=True,
                with_stack=True,
                with_modules=True,
            ):
                for _ in range(PROFILE_ITERS):
                    _ = moe.forward(tokens, use_static_shape=True)
            if global_rank == 0:
                logging.info(f"Finished run {name} successfully")

        run_profiling(baseline_moe, "baseline")
        run_profiling(shuffling_moe, "shuffling")

    except Exception as e:
        logging.info(f"Failed to run tests due to {e}")
        logging.error(traceback.format_exc())

    torch.distributed.destroy_process_group()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Arguments for testing MetaShuffling MoE."
    )
    parser.add_argument("--dim", type=int, default=5120)
    parser.add_argument("--hidden-dim", type=int, default=16384)
    parser.add_argument("--mp-size", type=int, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--num-experts", "-e", type=int, required=True)
    parser.add_argument("--mp-size-for-routed-experts", type=int)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--use-fast-accum", action="store_true", default=False)
    parser.add_argument("--dedup-comm", action="store_true", default=False)
    args = parser.parse_args()
    moe_args = MoEArgs(**vars(args))
    with tempfile.TemporaryDirectory():
        logging.info("Launching demo in parent process.")
        launcher.elastic_launch(get_launch_config(), entrypoint=run_demo)(moe_args)


if __name__ == "__main__":
    main()
