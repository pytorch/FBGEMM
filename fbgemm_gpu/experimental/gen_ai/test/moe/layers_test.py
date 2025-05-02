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
from fbgemm_gpu.experimental.gen_ai.moe.layers import MoE, MoEArgs, TokenShufflingMoE
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


def create_dummy_moe(args: MoEArgs) -> MoE:
    torch.set_default_dtype(torch.bfloat16)
    moe = MoE(args)
    moe.to("cuda")
    return moe


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
@pytest.mark.parametrize("dedup_all2all", [True, False])
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
    dedup_all2all: bool,
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

        moe = create_dummy_moe(args)

        token_shuffle_moe = TokenShufflingMoE(
            feed_forward=moe,
            ep_group=get_ep_group(),
            ep_mp_group=get_routed_experts_mp_group(),
            moe_args=args,
        )
        tokens = torch.randn(
            size=(32, 1, args.dim), device="cuda", dtype=torch.bfloat16
        )
        timestamp = datetime.timestamp(datetime.now())
        trace_filename = f"bench_token_shuffle_moe_{timestamp}_{global_rank}.json"
        for _ in range(WARM_UP_ITERS):
            _ = token_shuffle_moe.forward(tokens, use_static_shape=True)
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
                _ = token_shuffle_moe.forward(tokens, use_static_shape=True)
        if global_rank == 0:
            logging.info("Finished run successfully")
    except Exception as e:
        logging.info(f"Failed to run tests due to {e}")
        logging.error(traceback.format_exc())
    torch.distributed.destroy_process_group()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Arguments for attention benchmarks in Context Parallelism."
    )
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=16384)
    parser.add_argument("--ffn-dim-multiplier", type=float, default=1.2)
    parser.add_argument("--multiple-of", type=int, default=2048)
    parser.add_argument("--mp-size", type=int, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--num-experts", "-e", type=int, required=True)
    parser.add_argument("--mp-size-for-routed-experts", type=int)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--auto-scale-F", action="store_true", default=False)
    parser.add_argument("--use-fast-accum", action="store_true", default=False)
    parser.add_argument("--dedup-all2all", action="store_true", default=False)
    args = parser.parse_args()
    moe_args = MoEArgs(**vars(args))
    with tempfile.TemporaryDirectory():
        launcher.elastic_launch(get_launch_config(), entrypoint=run_demo)(moe_args)


if __name__ == "__main__":
    main()
