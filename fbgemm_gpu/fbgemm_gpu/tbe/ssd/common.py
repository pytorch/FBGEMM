#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import torch

# fmt:skip
from fbgemm_gpu.utils.loader import load_torch_module

try:
    load_torch_module(
        "//deeplearning/fbgemm/fbgemm_gpu:ssd_split_table_batched_embeddings"
    )
except Exception:
    pass

# Match cache associativity to the active hardware warp/wavefront width.  Newer
# AMD architectures such as gfx1250 use wave32, so the presence of a HIP build
# alone is not enough to select the right value.  Preserve the historical
# wave64 fallback when a ROCm build is imported without an available device.
ASSOC: int = (
    torch.cuda.get_device_properties(torch.cuda.current_device()).warp_size
    if torch.cuda.is_available()
    else (64 if torch.version.hip is not None else 32)
)


def pad4(value: int) -> int:
    """
    Compute the smallest multiple of 4 that is greater than or equal to the given value.

    Parameters:
        value (int): The integer to align (must be non-negative).

    Returns:
        int: The aligned value.

    Raises:
        ValueError: If the input is negative.
        TypeError: If the input is not an integer.
    """
    return (int(value) + 3) & ~3


def tensor_pad4(value: torch.Tensor) -> torch.Tensor:
    """
    The equivalent of pad4 for tensors.
    """
    return (value + 3) & ~3
