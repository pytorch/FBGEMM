#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

# From https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
# 1. cudaMemAdviseSetReadMostly = 1
# Data will mostly be read and only occassionally be written to
# 2. cudaMemAdviseUnsetReadMostly = 2
# Undo the effect of cudaMemAdviseSetReadMostly
# 3. cudaMemAdviseSetPreferredLocation = 3
# Set the preferred location for the data as the specified device
# 4. cudaMemAdviseUnsetPreferredLocation = 4
# Clear the preferred location for the data
# 5. cudaMemAdviseSetAccessedBy = 5
# Data will be accessed by the specified device, so prevent page faults as much as possible
# 6. cudaMemAdviseUnsetAccessedBy = 6
# Let the Unified Memory subsystem decide on the page faulting policy for the specified device


# Apply cuda memory advise on t with the above cuda memory advise hints from device_t using cudaMemAdvise CUDA API
def cudaMemAdvise(
    t: torch.Tensor,
    advice: Enum,
    device_t: Optional[torch.Tensor] = None,
) -> None:
    torch.ops.fbgemm.cuda_mem_advise(t, advice.value, device_t)


# prefetch cuda memory from t to device_t using cudaMemPrefetchAsync CUDA API
def cudaMemPrefetchAsync(
    t: torch.Tensor,
    device_t: Optional[torch.Tensor] = None,
) -> None:
    torch.ops.fbgemm.cuda_mem_prefetch_async(t, device_t)
