# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import inspect
from typing import Any, Callable

import torch
import triton  # noqa: F401
from triton.testing import do_bench_cudagraph


# pyre-ignore
def do_bench_cudagraph_and_clear_cache(fn: Callable[[], Any]) -> float:
    # 1GB data. Enough to clear L2/L3 cache.
    cache: torch.Tensor = torch.empty(
        1024 * 1024 * 1024, device="cuda", dtype=torch.int8
    )

    # pyre-ignore
    def wrapped_fn() -> Any:
        cache.zero_()
        return fn()

    time_with_clear_cache = do_bench_cudagraph(wrapped_fn, rep=100)
    time_only_clear_cache = do_bench_cudagraph(lambda: cache.zero_(), rep=100)

    return time_with_clear_cache - time_only_clear_cache


# pyre-ignore
def name_test_func(fn, _, p) -> str:
    name = fn.__name__
    args = inspect.getfullargspec(fn).args
    if "target_fn" in p.kwargs:
        name = f"test_{p.kwargs['target_fn']}"
    for arg_name in args[1:]:
        if arg_name == "target_fn":
            continue
        name += f"_{arg_name}={p.kwargs[arg_name]}"
    return name
