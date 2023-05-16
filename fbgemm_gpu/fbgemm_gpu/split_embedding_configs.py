#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
from typing import Any, Dict  # noqa: F401

import torch


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
    MADGRAD = "madgrad"
    EXACT_ROWWISE_WEIGHTED_ADAGRAD = "exact_row_wise_weighted_adagrad"

    def __str__(self) -> str:
        return self.value


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


@enum.unique
class SparseType(enum.Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    BF16 = "bf16"

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
        else:
            raise ValueError(f"Unsupported sparse type: {ty}")

    def as_int(self) -> int:
        return {
            SparseType.FP32.value: 0,
            SparseType.FP16.value: 1,
            SparseType.INT8.value: 2,
            SparseType.INT4.value: 3,
            SparseType.INT2.value: 4,
            SparseType.BF16.value: 5,
            SparseType.FP8.value: 6,
        }[self.value]

    @staticmethod
    def from_dtype(dtype: torch.dtype) -> "SparseType":
        if dtype == torch.float32:
            return SparseType("fp32")
        elif dtype == torch.float16:
            return SparseType("fp16")
        elif dtype == torch.int8 or dtype == torch.uint8:
            return SparseType("int8")
        elif dtype == torch.quint4x2:
            return SparseType("int4")
        elif dtype == torch.quint2x4:
            return SparseType("int2")
        elif dtype == torch.bfloat16:
            return SparseType("bf16")
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
        }[self.value]

    def is_float(self) -> bool:
        if (
            self.value == SparseType.FP32.value
            or self.value == SparseType.FP16.value
            or self.value == SparseType.FP8.value
            or self.value == SparseType.BF16.value
        ):
            return True
        else:
            return False

    def default_config(self) -> QuantizationConfig:
        if self.value == SparseType.FP8.value:
            return FP8QuantizationConfig(4, 7)
        else:
            return QuantizationConfig()


ELEMENT_SIZE: Dict[SparseType, int] = {
    SparseType.FP32: 4,
    SparseType.FP16: 2,
    SparseType.FP8: 1,
    SparseType.INT8: 1,
    SparseType.BF16: 2,
    # SparseType.INT4: 0.5,
}
