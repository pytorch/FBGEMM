# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
from datetime import timedelta
from typing import Optional

import torch

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_group,
    get_model_parallel_world_size,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from pyre_extensions import none_throws
from torch.distributed import ProcessGroup

_ROUTED_EXPERTS_MP_GROUP: Optional[ProcessGroup] = None

_EP_GROUP: Optional[ProcessGroup] = None


@functools.lru_cache
def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


@functools.lru_cache
def get_global_rank() -> int:
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else int(os.environ.get("RANK", "0"))
    )


@functools.lru_cache
def get_ep_group() -> ProcessGroup:
    global _EP_GROUP  # noqa: F824
    return none_throws(_EP_GROUP)


def get_routed_experts_mp_group() -> ProcessGroup:
    global _ROUTED_EXPERTS_MP_GROUP  # noqa: F824
    return none_throws(_ROUTED_EXPERTS_MP_GROUP)


def is_torch_run() -> bool:
    return os.environ.get("TORCHELASTIC_RUN_ID") is not None


def get_master_port() -> int:
    if is_torch_run():
        return int(os.environ["MASTER_PORT"])
    return 0


def get_master_addr() -> str:
    if is_torch_run():
        return os.environ["MASTER_ADDR"]
    return "127.0.0.1"


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def init_torch_distributed(
    backend: str = "cpu:gloo,cuda:nccl",
    timeout: Optional[timedelta] = None,
) -> None:
    if (port := get_master_port()) != 0:
        os.environ["RANK"] = str(get_global_rank())
        os.environ["WORLD_SIZE"] = str(get_world_size())
        os.environ["MASTER_ADDR"] = get_master_addr()
        os.environ["MASTER_PORT"] = str(port)
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend=backend, timeout=timeout)


# The following is probably only working for a single host situation
def init_parallel(
    model_parallel_size: int,
    data_parallel_size: int,
    mp_size_for_routed_experts: Optional[int] = None,
    backend: str = "cpu:gloo,cuda:nccl",
    timeout: Optional[timedelta] = None,
) -> None:
    if not torch.distributed.is_initialized():
        init_torch_distributed(backend=backend, timeout=timeout)

    if not model_parallel_is_initialized():
        assert (
            get_world_size() == model_parallel_size * data_parallel_size
        ), f"world size must be equal to mp*dp, but got {get_world_size()} != {model_parallel_size} * {data_parallel_size}"
        initialize_model_parallel(model_parallel_size)

    global_rank = get_global_rank()
    global _ROUTED_EXPERTS_MP_GROUP
    assert _ROUTED_EXPERTS_MP_GROUP is None
    ranks = []
    if (
        get_model_parallel_world_size() == mp_size_for_routed_experts
        or mp_size_for_routed_experts is None
    ):
        _ROUTED_EXPERTS_MP_GROUP = get_model_parallel_group()
    else:
        for base_rank in range(
            0, get_world_size(), none_throws(mp_size_for_routed_experts)
        ):
            ranks = list(range(base_rank, base_rank + mp_size_for_routed_experts))
            group = torch.distributed.new_group(ranks, timeout=timeout)
            if global_rank in ranks:
                _ROUTED_EXPERTS_MP_GROUP = group

    global _EP_GROUP
    assert _EP_GROUP is None

    ranks = []
    num_ep_groups = get_world_size() // data_parallel_size
    for i in range(num_ep_groups):
        ranks = list(range(i, get_world_size(), num_ep_groups))
        group = torch.distributed.new_group(ranks, timeout=timeout)
        if global_rank in ranks:
            _EP_GROUP = group
