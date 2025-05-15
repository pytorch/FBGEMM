# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import argparse
import itertools
import logging
import os
import tempfile
import traceback
from datetime import datetime
from functools import partial
from typing import Tuple

import torch

if torch.cuda.is_available():
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
        triton_quantize_fp8_row,
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

from .parallelism import (
    get_ep_group,
    get_global_rank,
    get_routed_experts_mp_group,
    init_parallel,
)

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


def run_demo(cmd_args: argparse.Namespace) -> None:
    kwargs = dict(vars(cmd_args))
    use_static_shape = kwargs.pop("use_static_shape")
    profiling = kwargs.pop("profiling")
    kwargs.pop("testing")

    moe_args = MoEArgs(**kwargs)
    try:
        init_parallel(
            moe_args.mp_size,
            moe_args.ep_size,
            moe_args.mp_size_for_routed_experts,
        )

        global_rank = get_global_rank()
        logging.info(f"Running demo in child process at {global_rank=}.")

        ep_group = get_ep_group()
        mp_group = get_routed_experts_mp_group()

        torch.set_default_dtype(torch.bfloat16)
        torch.random.manual_seed(global_rank)

        def default_init_method(x: torch.Tensor) -> torch.Tensor:
            torch.nn.init.kaiming_uniform_(x)
            return x

        def fp8_rowwise_init_method(
            x: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            default_init_method(x)
            if x.ndim == 3:
                E, K, N = x.shape
                x = x.transpose(1, 2)
                x = x.reshape(-1, K)
                xq, xs = triton_quantize_fp8_row(x.cuda())
                return xq.reshape(E, N, K).transpose(1, 2), xs.reshape(E, N)
            else:
                assert x.ndim == 2
                xq, xs = triton_quantize_fp8_row(x.cuda())
                return xq, xs

        param_names = (
            "moe_w_in_eDF",
            "moe_w_out_eFD",
            "moe_w_swiglu_eDF",
            "w_in_shared_FD",
            "w_out_shared_DF",
            "w_swiglu_FD",
            "router_DE",
        )

        if moe_args.precision == "bf16":
            init_methods = {name: default_init_method for name in param_names}
        else:
            assert moe_args.precision == "fp8_rowwise"
            init_methods = {name: fp8_rowwise_init_method for name in param_names}
            # pyre-ignore[6]
            init_methods["router_DE"] = default_init_method

        baseline_moe = BaselineMoE(
            ep_group=ep_group,
            ep_mp_group=mp_group,
            moe_args=moe_args,
        ).build(init_methods)
        baseline_moe.to("cuda")

        torch.random.manual_seed(global_rank)
        shuffling_moe = MetaShufflingMoE(
            ep_group=ep_group,
            ep_mp_group=mp_group,
            moe_args=moe_args,
        ).build(init_methods)
        shuffling_moe.to("cuda")

        tokens = torch.empty(
            size=(32, 32, moe_args.dim), device="cuda", dtype=torch.bfloat16
        )
        torch.nn.init.trunc_normal_(tokens, std=0.1, a=-0.2, b=0.2)

        # SharedExperts MP is the same as RoutedExperts MP.
        torch.distributed.broadcast(
            tokens,
            src=torch.distributed.get_global_rank(mp_group, 0),
            group=mp_group,
        )

        baseline_output = baseline_moe(tokens, use_static_shape=use_static_shape)
        torch.distributed.all_reduce(
            baseline_output,
            group=mp_group,
        )

        shuffling_output = shuffling_moe(tokens, use_static_shape=use_static_shape)
        torch.distributed.all_reduce(
            shuffling_output,
            group=mp_group,
        )

        logging.info(
            f"{global_rank=}, {baseline_output.norm()=}, {shuffling_output.norm()=}"
        )

        if moe_args.precision == "bf16":
            atol = 2e-3
            rtol = 1.6e-2
        else:
            atol = 4e-3
            rtol = 1.6e-2
        torch.testing.assert_close(
            baseline_output,
            shuffling_output,
            atol=atol,
            rtol=rtol,
        )

        if profiling:

            def run_profiling(moe: torch.nn.Module, name: str):
                timestamp = datetime.timestamp(datetime.now())
                trace_filename = f"bench_{name}_{timestamp}_{global_rank}.json"
                for _ in range(WARM_UP_ITERS):
                    _ = moe.forward(tokens, use_static_shape=use_static_shape)
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
                        _ = moe.forward(tokens, use_static_shape=use_static_shape)

            run_profiling(baseline_moe, "baseline")
            run_profiling(shuffling_moe, "shuffling")
        logging.info(f"Successed to run demo with {cmd_args}!")

    except Exception as e:
        logging.info(f"Failed to run demo with {cmd_args}! Reason: {e}.")
        logging.info(traceback.format_exc())

    torch.distributed.destroy_process_group()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Arguments for testing MetaShuffling MoE."
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp8_rowwise"],
    )
    parser.add_argument("--dim", type=int, default=5120)
    parser.add_argument("--hidden-dim", type=int, default=16384)
    parser.add_argument("--num-experts", "-e", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--mp-size", type=int, default=4)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--mp-size-for-routed-experts", type=int)
    parser.add_argument("--use-static-shape", type=bool, default=False)
    parser.add_argument("--dedup-comm", type=bool, default=False)
    parser.add_argument("--use-fast-accum", type=bool, default=False)

    parser.add_argument("--testing", "-t", action="store_true", default=False)
    parser.add_argument("--profiling", "-p", action="store_true", default=False)

    args = parser.parse_args()
    if args.testing:
        setting_dict = {
            "precision": ["bf16", "fp8_rowwise"],
            "dim": [5120],
            "hidden_dim": [16384],
            "num_experts": [128],
            "top_k": [1],
            "ep_size": [1, 2],
            "use_static_shape": [True, False],
            "dedup_comm": [True, False],
            "use_fast_accum": [True],
        }
        for setting in itertools.product(*setting_dict.values()):
            for name, value in zip(setting_dict.keys(), setting):
                setattr(args, name, value)
            # TODO: Cleanup this hardcoded setting.
            world_size = 8
            args.mp_size = world_size // args.ep_size
            args.mp_size_for_routed_experts = args.mp_size

            with tempfile.TemporaryDirectory():
                launcher.elastic_launch(get_launch_config(), entrypoint=run_demo)(args)

            break
    else:
        with tempfile.TemporaryDirectory():
            launcher.elastic_launch(get_launch_config(), entrypoint=run_demo)(args)


if __name__ == "__main__":
    main()
