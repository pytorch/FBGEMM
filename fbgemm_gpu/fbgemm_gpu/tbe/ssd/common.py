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

# Preserve the historical value for callers that only use ASSOC as a static
# compatibility constant.  Device-bound SSD operators must call
# get_cache_assoc() instead: gfx1250 is wave32 even though older ROCm devices
# are wave64, and querying the current device while importing this module makes
# the result stale on heterogeneous hosts.
ASSOC: int = 64 if torch.version.hip is not None else 32


def get_cache_assoc(device: torch.device | int | None = None) -> int:
    """Return the cache associativity for ``device`` without import-time I/O."""
    if isinstance(device, torch.device) and device.type == "cpu":
        return ASSOC
    if not torch.cuda.is_available():
        if device is not None:
            raise RuntimeError(
                "GPU runtime must be available to determine device cache associativity"
            )
        return ASSOC
    selected_device = torch.cuda.current_device() if device is None else device
    return int(torch.cuda.get_device_properties(selected_device).warp_size)


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
