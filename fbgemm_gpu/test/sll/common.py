# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import fbgemm_gpu
import fbgemm_gpu.sll
import torch

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if not open_source:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")


def clone_tensor(data: torch.Tensor) -> torch.Tensor:
    if data.requires_grad:
        return data.detach().clone().requires_grad_()
    return data.detach().clone()
