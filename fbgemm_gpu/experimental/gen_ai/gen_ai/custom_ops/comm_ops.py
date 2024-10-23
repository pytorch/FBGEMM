# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import torch

from .load_custom_library import load_custom_library

"""
This file contains manual shape registrations for communication custom operators.
These are needed for custom operators to be compatible with torch.compile.

In some cases, fake tensor handling can be done by registering a meta implementation
directly in cpp. However, for more complicated functions such as those that involve
cross device synchronization, pytorch requires a full fake implementation be registered
in python.
"""

# Load all custom operators.
load_custom_library("//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:comm_ops")


@torch.library.register_fake("fbgemm::nccl_allreduce")
def nccl_allreduce_abstract(
    dst: torch.Tensor,
    src: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    comm_idx: int = 0,
) -> None:
    return None


@torch.library.register_fake("fbgemm::nccl_allgather")
def nccl_allgather_abstract(
    dst: torch.Tensor,
    src: torch.Tensor,
    comm_idx: int = 0,
) -> None:
    return None


@torch.library.register_fake("fbgemm::nccl_alltoall")
def nccl_alltoall_abstract(
    dst: torch.Tensor,
    src: torch.Tensor,
    world_size: int,
    comm_idx: int = 0,
) -> None:
    return None


@torch.library.register_fake("fbgemm::nccl_reducescatter")
def nccl_reducescatter_abstract(
    dst: torch.Tensor,
    src: torch.Tensor,
    comm_idx: int = 0,
) -> None:
    return None


@torch.library.register_fake("fbgemm::one_shot_car_allreduce")
def one_shot_car_allreduce_abstract(
    dst: torch.Tensor,
    src: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    comm_idx: int = 0,
) -> None:
    return None


@torch.library.register_fake("fbgemm::two_shot_car_allreduce")
def two_shot_car_allreduce_abstract(
    dst: torch.Tensor,
    src: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    comm_idx: int = 0,
) -> None:
    return None
