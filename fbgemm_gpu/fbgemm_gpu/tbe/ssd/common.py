#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

from collections.abc import Callable

import torch

# fmt:skip
from fbgemm_gpu.utils.loader import load_torch_module

try:
    load_torch_module(
        "//deeplearning/fbgemm/fbgemm_gpu:ssd_split_table_batched_embeddings"
    )
except Exception:
    pass

# Legacy default for offline capacity estimates. Live SSD caches query the
# device warp size so wave32 and wave64 ROCm devices use matching layouts.
ASSOC: int = 32 if torch.version.hip is None else 64


def get_cache_assoc(device: torch.device | int | None = None) -> int:
    if device is None:
        if not torch.cuda.is_available():
            return ASSOC
        device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).warp_size


def normalize_device(device: torch.device | int | None) -> torch.device:
    if device is None or isinstance(device, int):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU runtime must be available to construct SSD TBE")
        index = torch.cuda.current_device() if device is None else device
        return torch.device(torch.accelerator.current_accelerator().type, index)
    if device.type == "cpu" or device.type == "meta":
        raise ValueError("SSD TBE requires a GPU device")
    if device.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU runtime must be available to construct SSD TBE")
        return torch.device(device.type, torch.cuda.current_device())
    return device


def ensure_same_device(
    current_device: torch.device, fn: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    target_device = fn(torch.empty(0, device=current_device)).device
    if target_device != current_device:
        raise RuntimeError(
            "SSD TBE cannot be moved after construction because its CUDA streams, "
            "events, and SSD backend are device-bound; pass device= to the "
            "constructor instead"
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
