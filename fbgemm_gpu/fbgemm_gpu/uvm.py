#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from enum import Enum
from typing import Optional

import torch

from fbgemm_gpu.enums import create_enums

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:cumem_utils")

# Import all uvm enums from c++ library
create_enums(globals(), torch.ops.fbgemm.fbgemm_gpu_uvm_enum_query)


def cudaMemAdvise(
    t: torch.Tensor,
    advice: Enum,
) -> None:
    torch.ops.fbgemm.cuda_mem_advise(t, advice.value)


def cudaMemPrefetchAsync(
    t: torch.Tensor,
    device_t: Optional[torch.Tensor] = None,
) -> None:
    torch.ops.fbgemm.cuda_mem_prefetch_async(t, device_t)
