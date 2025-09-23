#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum
import itertools
from typing import Any, Dict  # noqa: F401

import torch

from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    SplitState,
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


def pad16(value: int) -> int:
    """
    Compute the smallest multiple of 16 that is greater than or equal to the given value.

    Parameters:
        value (int): The integer to align (must be non-negative).

    Returns:
        int: The aligned value.

    Raises:
        ValueError: If the input is negative.
        TypeError: If the input is not an integer.
    """
    return (int(value) + 15) & ~15


@enum.unique
class EmbOptimType(enum.Enum):
    SGD = "sgd"  # uses non-deterministic updates (atomicAdd(..)) with duplicate ids
    EXACT_SGD = (
        "exact_sgd"  # uses deterministic updates (via sorting + segment reduction)
    )
    LAMB = "lamb"
    ADAM = "adam"
    # exact/dedup: gradients to the same row are applied with coalesce then apply
    # together, instead of applied in sequence (approx).
    EXACT_ADAGRAD = "exact_adagrad"
    EXACT_ROWWISE_ADAGRAD = "exact_row_wise_adagrad"
    LARS_SGD = "lars_sgd"
    PARTIAL_ROWWISE_ADAM = "partial_row_wise_adam"
    PARTIAL_ROWWISE_LAMB = "partial_row_wise_lamb"
    ROWWISE_ADAGRAD = "row_wise_adagrad"
    SHAMPOO = "shampoo"  # not currently supported for sparse embedding tables
    SHAMPOO_V2 = "shampoo_v2"  # not currently supported for sparse embedding tables
    MADGRAD = "madgrad"
    EXACT_ROWWISE_WEIGHTED_ADAGRAD = "exact_row_wise_weighted_adagrad"  # deprecated
    ENSEMBLE_ROWWISE_ADAGRAD = "ensemble_row_wise_adagrad"
    EMAINPLACE_ROWWISE_ADAGRAD = "ema_in_place_row_wise_adagrad"
    NONE = "none"

    def __str__(self) -> str:
        return self.value

    def _extract_dtype(
        self, optimizer_state_dtypes: dict[str, "SparseType"], name: str
    ) -> torch.dtype:
        if optimizer_state_dtypes is None or name not in optimizer_state_dtypes:
            return torch.float32
        return optimizer_state_dtypes[name].as_dtype()

    def state_names(self) -> list[str]:
        """
        Returns the names of the optimizer states.  The order of the states will
        be the order in which they are processed and returned in
        SSDTableBatchedEmbeddingBags.split_optimizer_states(), but this is not
        necessarily the same as the order they are stored in the memory layout.
        """
        if self == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
            return ["momentum1"]
        elif self in [EmbOptimType.PARTIAL_ROWWISE_ADAM, EmbOptimType.ADAM]:
            return ["momentum1", "momentum2"]
        else:
            return []

    def state_size_table(self, D: int) -> dict[str, int]:
        """
        Returns the table of state names to state sizes in terms of number of
        elements (per table row)
        """
        if self == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
            return {"momentum1": 1}
        elif self == EmbOptimType.PARTIAL_ROWWISE_ADAM:
            return {"momentum1": D, "momentum2": 1}
        elif self == EmbOptimType.ADAM:
            return {"momentum1": D, "momentum2": D}
        else:
            return {}

    def state_size_nbytes(
        self,
        D: int,
        optimizer_state_dtypes: dict[str, "SparseType"] = {},  # noqa: B006
    ) -> int:
        """
        Returns the size of the data (in bytes) required to hold the optimizer
        state (per table row).  This size includes byte-padding.
        """
        momentum1_dtype = self._extract_dtype(optimizer_state_dtypes, "momentum1")
        momentum2_dtype = self._extract_dtype(optimizer_state_dtypes, "momentum2")

        if self == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
            return momentum1_dtype.itemsize

        elif self == EmbOptimType.PARTIAL_ROWWISE_ADAM:
            return pad4(1 * momentum2_dtype.itemsize) + D * momentum1_dtype.itemsize

        elif self == EmbOptimType.ADAM:
            return (D * momentum1_dtype.itemsize) + (D * momentum2_dtype.itemsize)

        else:
            return 0

    def byte_offsets_along_row(
        self,
        D: int,
        weights_precision: "SparseType",
        optimizer_state_dtypes: dict[str, "SparseType"] = {},  # noqa: B006
    ) -> dict[str, tuple[int, int]]:
        """
        Returns the start and end byte offsets of each optimizer state along a
        cache row with optimizer state offloading enabled.
        """
        # Extract the optimizer state dtypes
        momentum1_dtype = self._extract_dtype(optimizer_state_dtypes, "momentum1")
        momentum2_dtype = self._extract_dtype(optimizer_state_dtypes, "momentum2")

        # This is the pointer to where the optimizer state begins in the memory
        p0 = pad4(D) * weights_precision.as_dtype().itemsize

        if self == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
            return {"momentum1": (p0, p0 + momentum1_dtype.itemsize)}

        elif self == EmbOptimType.PARTIAL_ROWWISE_ADAM:
            # momentum1 lies after momentum2
            p1 = p0 + pad4(1 * momentum2_dtype.itemsize)
            return {
                "momentum2": (p0, p0 + momentum2_dtype.itemsize),
                "momentum1": (
                    p1,
                    p1 + D * momentum1_dtype.itemsize,
                ),
            }

        elif self == EmbOptimType.ADAM:
            # momentum2 lies after momentum1
            p1 = p0 + (D * momentum1_dtype.itemsize)

            return {
                "momentum1": (p0, p1),
                "momentum2": (p1, p1 + D * momentum2_dtype.itemsize),
            }

        else:
            return {}

    def empty_states(
        self,
        rows: list[int],
        dims: list[int],
        optimizer_state_dtypes: dict[str, "SparseType"] = {},  # noqa: B006
    ) -> list[list[torch.Tensor]]:
        """
        Creates sets of empty tensors per table to hold optimizer states based
        on the specified optimizer type, state dtypes, embedding specs, and
        (optionally) local row counts.
        """
        # Else, check that the local row count for each table is set
        assert len(rows) == len(dims)

        opt_states_set: list[list[torch.Tensor]] = []

        for r, D in zip(rows, dims):
            # Set up the table of state names to state sizes, ordered by their
            # memory layout
            state_size_table = self.state_size_table(D)
            ordered_state_sizes = [(k, state_size_table[k]) for k in self.state_names()]

            # Create the optimizer states for this table
            opt_states_set.append(
                [
                    torch.empty(
                        # If the state size is 1, then fix tensor to 1D to be
                        # consistent with training.py code
                        # pyre-ignore [6]
                        (r, d) if d > 1 else r,
                        dtype=self._extract_dtype(optimizer_state_dtypes, state_name),
                        device="cpu",
                    )
                    for state_name, d in ordered_state_sizes
                ]
            )

        return opt_states_set

    def ssd_state_splits(
        self,
        embedding_specs: list[tuple[int, int]],  # Tuple of (rows, dims)
        optimizer_state_dtypes: dict[str, "SparseType"] = {},  # noqa: B006
        enable_optimizer_offloading: bool = False,
    ) -> list[tuple[SplitState, str, torch.dtype]]:
        """
        Returns the split planning for the optimizer states
        """
        (rows, _) = zip(*embedding_specs)
        T_ = len(embedding_specs)

        # This is the cumulative row counts for rowwise states
        row_count_cumsum: list[int] = [0] + list(itertools.accumulate(rows))
        # This is the cumulative element counts for elementwise states
        table_size_cumsum: list[int] = [0] + list(
            itertools.accumulate([r * d for r, d in embedding_specs])
        )

        if self == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
            params = {"momentum1": row_count_cumsum}
        elif self == EmbOptimType.PARTIAL_ROWWISE_ADAM:
            params = {"momentum1": table_size_cumsum, "momentum2": row_count_cumsum}
        elif self == EmbOptimType.ADAM:
            params = {
                "momentum1": table_size_cumsum,
                "momentum2": table_size_cumsum,
                "row_counter": row_count_cumsum,
            }
        else:
            params = {}

        return [
            (
                SplitState(
                    dev_size=(
                        cumsum_table[-1] if not enable_optimizer_offloading else 0
                    ),
                    host_size=0,
                    uvm_size=0,
                    placements=[EmbeddingLocation.DEVICE for _ in range(T_)],
                    offsets=cumsum_table[:-1],
                ),
                name,
                self._extract_dtype(optimizer_state_dtypes, name),
            )
            for (name, cumsum_table) in params.items()
        ]


# Base class for quantization configuration (in case other numeric types have
# configs)
class QuantizationConfig:
    def __init__(self) -> None:
        self.config = {}  # type: Dict[str, Any]

    def get(self, name: str) -> int:
        return -1


# FP8 quantization configuration
# Compute necessary parameters in the constructor
class FP8QuantizationConfig(QuantizationConfig):
    def __init__(self, exponent_bits: int, exponent_bias: int) -> None:
        super(FP8QuantizationConfig, self).__init__()
        self.config = {
            "exponent_bits": exponent_bits,
            "exponent_bias": exponent_bias,
            "max_position": (1 << ((1 << exponent_bits) - 2 - exponent_bias))
            * (2 - 2 ** (exponent_bits - 7)),
        }  # type: Dict[str, Any]

    def get(self, name: str) -> int:
        if name not in self.config:
            raise RuntimeError("{} must be set in config".format(name))
        return self.config[name]


def sparse_type_to_int(sparse_type: "SparseType") -> int:
    return {
        SparseType.FP32.value: 0,
        SparseType.FP16.value: 1,
        SparseType.INT8.value: 2,
        SparseType.INT4.value: 3,
        SparseType.INT2.value: 4,
        SparseType.BF16.value: 5,
        SparseType.FP8.value: 6,
        SparseType.MX4.value: 7,
        SparseType.NFP8.value: 8,
    }[sparse_type.value]


@enum.unique
class SparseType(enum.Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    FP8 = "fp8"
    # NFP8 refers to "native" FP8 in that it uses the GPU implementations
    # of E4M3 whereas the other FP8 sparsetype uses a custom format. Use of
    # NFP8 allows us to use hardware casting intrinsics which can be much faster.
    # Eventually, we should merge these two types.
    NFP8 = "nfp8"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    BF16 = "bf16"
    MX4 = "mx4"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_int(ty: int) -> "SparseType":
        if ty == 0:
            return SparseType("fp32")
        elif ty == 1:
            return SparseType("fp16")
        elif ty == 2:
            return SparseType("int8")
        elif ty == 3:
            return SparseType("int4")
        elif ty == 4:
            return SparseType("int2")
        elif ty == 5:
            return SparseType("bf16")
        elif ty == 6:
            return SparseType("fp8")
        elif ty == 8:
            return SparseType("mx4")
        elif ty == 9:
            return SparseType("nfp8")
        else:  # Invalid is 7 or non enumerated.
            raise ValueError(f"Unsupported sparse type: {ty}")

    def as_int(self) -> int:
        return sparse_type_to_int(self)

    @staticmethod
    def from_dtype(dtype: torch.dtype, is_mx: bool = False) -> "SparseType":
        if dtype == torch.float32:
            return SparseType("fp32")
        elif dtype == torch.float16:
            return SparseType("fp16")
        elif (dtype == torch.int8 or dtype == torch.uint8) and not is_mx:
            return SparseType("int8")
        elif dtype == torch.quint4x2:
            return SparseType("int4")
        elif dtype == torch.quint2x4:
            return SparseType("int2")
        elif dtype == torch.bfloat16:
            return SparseType("bf16")
        elif dtype == torch.uint8:
            return SparseType("mx4")
        elif dtype == torch.float8_e4m3fnuz or dtype == torch.float8_e4m3fn:
            return SparseType("nfp8")
        else:
            raise ValueError(f"Unsupported sparse dtype: {dtype}")

    def as_dtype(self) -> torch.dtype:
        return {
            SparseType.FP32.value: torch.float32,
            SparseType.FP16.value: torch.float16,
            SparseType.FP8.value: torch.uint8,
            SparseType.INT8.value: torch.uint8,
            SparseType.INT4.value: torch.quint4x2,
            SparseType.INT2.value: torch.quint2x4,
            SparseType.BF16.value: torch.bfloat16,
            SparseType.MX4.value: torch.uint8,
            SparseType.NFP8.value: (
                torch.float8_e4m3fnuz
                if torch.version.hip is not None
                else torch.float8_e4m3fn
            ),
        }[self.value]

    def bit_rate(self) -> int:
        return {
            SparseType.FP32.value: 32,
            SparseType.FP16.value: 16,
            SparseType.FP8.value: 8,
            SparseType.INT8.value: 8,
            SparseType.INT4.value: 4,
            SparseType.INT2.value: 2,
            SparseType.BF16.value: 16,
            SparseType.MX4.value: 4,
            SparseType.NFP8.value: 8,
        }[self.value]

    def align_size(self) -> int:
        return {
            SparseType.FP32.value: 1,
            SparseType.FP16.value: 2,
            SparseType.FP8.value: 4,
            SparseType.INT8.value: 4,
            SparseType.INT4.value: 8,
            SparseType.INT2.value: 16,
            SparseType.BF16.value: 2,
            SparseType.MX4.value: 8,
            SparseType.NFP8.value: 4,
        }[self.value]

    def is_float(self) -> bool:
        if (
            self.value == SparseType.FP32.value
            or self.value == SparseType.FP16.value
            or self.value == SparseType.FP8.value
            or self.value == SparseType.BF16.value
            or self.value == SparseType.NFP8.value
        ):
            return True
        else:
            return False

    def default_config(self) -> QuantizationConfig:
        if self.value == SparseType.FP8.value:
            return FP8QuantizationConfig(4, 7)
        else:
            return QuantizationConfig()


ELEMENT_SIZE: dict[SparseType, int] = {
    SparseType.FP32: 4,
    SparseType.FP16: 2,
    SparseType.FP8: 1,
    SparseType.INT8: 1,
    SparseType.BF16: 2,
    SparseType.NFP8: 1,
    # SparseType.INT4: 0.5,
}
