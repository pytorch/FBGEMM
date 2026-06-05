#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Foundational embedding, compute, and pooling type definitions for TBE.

This module contains IntEnums, NamedTuples, constants, and utility functions
that form the base vocabulary for Split Table Batched Embeddings (TBE).
"""

import enum
from typing import NamedTuple

import torch
from torch import Tensor

# Maximum number of times prefetch() can be called without
# a corresponding forward() call
MAX_PREFETCH_DEPTH: int = 100

# GPU and CPU use 16-bit scale and bias for quantized embedding bags in TBE
# The total size is 2 + 2 = 4 bytes
DEFAULT_SCALE_BIAS_SIZE_IN_BYTES: int = 4

# Row dimension offset for INT8 quantized embeddings
INT8_EMB_ROW_DIM_OFFSET: int = 8


class EmbeddingLocation(enum.IntEnum):
    DEVICE = 0
    MANAGED = 1
    MANAGED_CACHING = 2
    HOST = 3
    MTIA = 4

    @classmethod
    def str_values(cls) -> list[str]:
        return [
            "device",
            "managed",
            "managed_caching",
            "host",
            "mtia",
        ]

    @classmethod
    def from_str(cls, key: str) -> "EmbeddingLocation":
        lookup = {
            "device": EmbeddingLocation.DEVICE,
            "managed": EmbeddingLocation.MANAGED,
            "managed_caching": EmbeddingLocation.MANAGED_CACHING,
            "host": EmbeddingLocation.HOST,
            "mtia": EmbeddingLocation.MTIA,
        }
        if key in lookup:
            return lookup[key]
        else:
            raise ValueError(f"Cannot parse value into EmbeddingLocation: {key}")

    @classmethod
    def from_device_and_clf(
        cls, device: torch.device, cache_load_factor: float
    ) -> "EmbeddingLocation":
        """Determine embedding location from device type and cache load factor.

        Formerly the free function get_new_embedding_location() in
        split_table_batched_embeddings_ops_common.py.
        """
        # Only support CPU and GPU device
        assert device.type == "cpu" or device.type == "cuda"
        if cache_load_factor < 0 or cache_load_factor > 1:
            raise ValueError(
                f"cache_load_factor must be between 0.0 and 1.0, got {cache_load_factor}"
            )

        if device.type == "cpu":
            return EmbeddingLocation.HOST
        # UVM only
        elif cache_load_factor == 0:
            return EmbeddingLocation.MANAGED
        # HBM only
        elif cache_load_factor == 1.0:
            return EmbeddingLocation.DEVICE
        # UVM caching
        else:
            return EmbeddingLocation.MANAGED_CACHING


class PoolingMode(enum.IntEnum):
    SUM = 0
    MEAN = 1
    NONE = 2

    def do_pooling(self) -> bool:
        return self is not PoolingMode.NONE

    @classmethod
    def from_str(cls, key: str) -> "PoolingMode":
        lookup = {
            "sum": PoolingMode.SUM,
            "mean": PoolingMode.MEAN,
            "none": PoolingMode.NONE,
        }
        if key in lookup:
            return lookup[key]
        else:
            raise ValueError(f"Cannot parse value into PoolingMode: {key}")


class BoundsCheckMode(enum.IntEnum):
    # Raise an exception (CPU) or device-side assert (CUDA)
    FATAL = 0
    # Log the first out-of-bounds instance per kernel, and set to zero.
    WARNING = 1
    # Set to zero.
    IGNORE = 2
    # No bounds checks.
    NONE = 3
    # IGNORE with V2 enabled
    V2_IGNORE = 4
    # WARNING with V2 enabled
    V2_WARNING = 5
    # FATAL with V2 enabled
    V2_FATAL = 6


class ComputeDevice(enum.IntEnum):
    CPU = 0
    CUDA = 1
    MTIA = 2

    @classmethod
    def get_available(cls) -> "ComputeDevice":
        """Return the best available compute device.

        Formerly the free function get_available_compute_device() in
        split_table_batched_embeddings_ops_training.py.
        """
        if torch.cuda.is_available():
            return ComputeDevice.CUDA
        elif torch.mtia.is_available():
            return ComputeDevice.MTIA
        else:
            return ComputeDevice.CPU


class EmbeddingSpecInfo(enum.IntEnum):
    feature_names = 0
    rows = 1
    dims = 2
    sparse_type = 3
    embedding_location = 4


RecordCacheMetrics: NamedTuple = NamedTuple(
    "RecordCacheMetrics",
    [("record_cache_miss_counter", bool), ("record_tablewise_cache_miss", bool)],
)

SplitState: NamedTuple = NamedTuple(
    "SplitState",
    [
        ("dev_size", int),
        ("host_size", int),
        ("uvm_size", int),
        ("placements", list[EmbeddingLocation]),
        ("offsets", list[int]),
    ],
)


# NOTE: This is also defined in fbgemm_gpu.tbe.utils, but declaring
# target dependency on :split_embedding_utils will result in compatibility
# breakage with Caffe2 module_factory because it will pull in numpy
def round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def tensor_to_device(tensor: torch.Tensor, device: torch.device) -> Tensor:
    if tensor.device == torch.device("meta"):
        return torch.empty_like(tensor, device=device)
    return tensor.to(device)


def get_bounds_check_version_for_platform() -> int:
    # NOTE: Use bounds_check_indices v2 on ROCm because ROCm has a
    # constraint that the gridDim * blockDim has to be smaller than
    # 2^32. The v1 kernel can be launched with gridDim * blockDim >
    # 2^32 while the v2 kernel limits the gridDim size to 64 * # of
    # SMs.  Thus, its gridDim * blockDim is guaranteed to be smaller
    # than 2^32
    return 2 if (torch.cuda.is_available() and torch.version.hip) else 1
