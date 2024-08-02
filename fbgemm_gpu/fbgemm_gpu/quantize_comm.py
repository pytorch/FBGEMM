#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# The code in this file is refactored from https://fburl.com/code/p2gy2gxb
# based on "Amy Yang et al., Training Deep Learning Recommendation Model with
# Quantized Collective Communications", DLP-KDD 2020.


import logging
from typing import Optional, TypeVar

import torch

from fbgemm_gpu.quantize_utils import (
    bf16_to_fp32,
    fp16_to_fp32,
    fp32_to_bf16_with_clamp,
    fp32_to_fp16_with_clamp,
    fp32_to_hfp8_with_clamp,
    fp32_to_mx4,
    hfp8_to_fp32,
    mx4_to_fp32,
    RoundingMode,
)

from fbgemm_gpu.split_embedding_configs import SparseType

from torch.autograd.profiler import record_function  # usort:skip
from dataclasses import dataclass

import fbgemm_gpu.quantize.quantize_ops  # noqa F401

logger: logging.Logger = logging.getLogger()

# FP8 configurations
ebits, mbits, bias = 4, 3, 15
max_pos: float = (2 ** ((1 << ebits) - 2 - bias)) * (2 - 2 ** (-mbits))

# INT8 configurations
ROW_DIM_DEFAULT = 32

# MX4 configurations
MX_GROUP_SIZE_DEFAULT = 32


def none_throws(
    optional: Optional[TypeVar("_T")], message: str = "Unexpected `None`"
) -> TypeVar("_T"):
    if optional is None:
        raise AssertionError(message)
    return optional


@dataclass
class QuantizationContext:
    row_dim: int = ROW_DIM_DEFAULT
    row_dim_quant: int = -1
    mx_group_size: int = MX_GROUP_SIZE_DEFAULT
    rounding_mode: RoundingMode = RoundingMode.ceil


def _quantize_tensor(
    input_tensor: torch.Tensor,
    comm_precision: SparseType,
    ctx: Optional[QuantizationContext] = None,
    is_fwd: bool = True,
) -> torch.Tensor:
    if comm_precision == SparseType.FP32:
        return input_tensor
    elif comm_precision == SparseType.FP16:
        return fp32_to_fp16_with_clamp(input_tensor)
    elif comm_precision == SparseType.BF16:
        return fp32_to_bf16_with_clamp(input_tensor)
    elif comm_precision == SparseType.FP8:
        # return fp32_to_hfp8_with_clamp(input_tensor, ebits, mbits, bias)
        if ctx is not None and ctx.row_dim > 0:
            ctx = none_throws(ctx)
            row_dim = ctx.row_dim
            input_2d = input_tensor.view((-1, row_dim)) if row_dim > 0 else input_tensor
            input_2d_quant = torch.ops.fbgemm.FloatToFP8RowwiseQuantized(
                input_2d, is_fwd
            )
            row_dim_quant = input_2d_quant.shape[1]
            input_quant_all2all = None
            input_quant_all2all = input_2d_quant.view((-1))
            ctx.row_dim_quant = row_dim_quant
            return input_quant_all2all
        else:
            return fp32_to_hfp8_with_clamp(input_tensor, ebits, mbits, bias)
    elif comm_precision == SparseType.INT8:
        ctx = none_throws(ctx)
        row_dim = ctx.row_dim
        input_2d = input_tensor.view((-1, row_dim)) if row_dim > 0 else input_tensor
        input_2d_quant = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_2d)
        row_dim_quant = input_2d_quant.shape[1]
        input_quant_all2all = None
        input_quant_all2all = input_2d_quant.view((-1))
        ctx.row_dim_quant = row_dim_quant
        return input_quant_all2all
    elif comm_precision == SparseType.MX4:
        mx_group_size = ctx.mx_group_size if ctx is not None else MX_GROUP_SIZE_DEFAULT
        rounding_mode = ctx.rounding_mode if ctx is not None else RoundingMode.ceil
        return fp32_to_mx4(input_tensor, mx_group_size, rounding_mode=rounding_mode)
    else:
        raise ValueError(f"comm_precision={comm_precision} is not supported")


def _dequantize_tensor(
    quantized_tensor: torch.Tensor,
    comm_precision: SparseType,
    ctx: Optional[QuantizationContext] = None,
    is_fwd: bool = True,
) -> torch.Tensor:
    if comm_precision == SparseType.FP32:
        assert quantized_tensor.dtype == torch.float
        return quantized_tensor
    elif comm_precision == SparseType.FP16:
        assert quantized_tensor.dtype == torch.half
        return fp16_to_fp32(quantized_tensor)
    elif comm_precision == SparseType.BF16:
        assert quantized_tensor.dtype == torch.bfloat16
        return bf16_to_fp32(quantized_tensor)
    elif comm_precision == SparseType.FP8:
        if ctx is not None and ctx.row_dim > 0:
            row_dim_quant = ctx.row_dim_quant
            quantized_tensor_2d = quantized_tensor.view((-1, row_dim_quant))
            dequant_tensor = torch.ops.fbgemm.FP8RowwiseQuantizedToFloat(
                quantized_tensor_2d, is_fwd
            )
            return dequant_tensor.view(-1)
        else:
            assert quantized_tensor.dtype == torch.uint8
            return hfp8_to_fp32(quantized_tensor, ebits, bias)
    elif comm_precision == SparseType.INT8:
        ctx = none_throws(ctx)
        row_dim_quant = ctx.row_dim_quant
        quantized_tensor_2d = quantized_tensor.view((-1, row_dim_quant))
        dequant_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
            quantized_tensor_2d
        )
        return dequant_tensor.view(-1)
    elif comm_precision == SparseType.MX4:
        mx_group_size = ctx.mx_group_size if ctx is not None else MX_GROUP_SIZE_DEFAULT
        return mx4_to_fp32(quantized_tensor, mx_group_size)
    else:
        raise ValueError(f"comm_precision={comm_precision} is not supported")


class QuantizedCommCodec:
    # Concrete implementation of QuantizedCommCodec provided by FBGEMM functions.
    def __init__(
        self,
        comm_precision: SparseType,
        loss_scale: Optional[float] = None,
        row_dim: Optional[int] = None,
        is_fwd: bool = True,
    ) -> None:
        if loss_scale is not None:
            if comm_precision not in [SparseType.FP16, SparseType.BF16]:
                logger.warning(
                    f"Setting loss scale for comm_precision={comm_precision} is not supported. Overriding to None"
                )
                loss_scale = None

        logger.info(
            f"Creating QuantizedCommsCodec comm_precision:{comm_precision}, loss_scale:{loss_scale}"
        )

        self._comm_precision = comm_precision
        self._loss_scale = loss_scale
        self._is_fwd = is_fwd
        self._row_dim: int = -1 if row_dim is None else row_dim

    def encode(
        self, input_tensor: torch.Tensor, ctx: Optional[QuantizationContext] = None
    ) -> torch.Tensor:
        if self._loss_scale is not None:
            input_tensor = self._loss_scale * input_tensor
        with record_function(
            f"## encoder {self._comm_precision} {self._loss_scale} ##"
        ):
            output = _quantize_tensor(
                input_tensor,
                self._comm_precision,
                ctx,
                self._is_fwd,
            )
        return output

    def decode(
        self, input_tensor: torch.Tensor, ctx: Optional[QuantizationContext] = None
    ) -> torch.Tensor:
        if self._loss_scale is not None:
            input_tensor = input_tensor / self._loss_scale
        with record_function(
            f"## decoder {self._comm_precision} {self._loss_scale} ##"
        ):
            dequantized_tensor = _dequantize_tensor(
                input_tensor, self._comm_precision, ctx, self._is_fwd
            )
        return dequantized_tensor

    def calc_quantized_size(
        self, input_len: int, ctx: Optional[QuantizationContext] = None
    ) -> int:
        # Use the same logic in _float_to_fused8bitrowwise_gpu_t()
        if self._comm_precision == SparseType.INT8 or (
            self._comm_precision == SparseType.FP8 and self._row_dim > 0
        ):
            ctx = none_throws(ctx)
            assert input_len % ctx.row_dim == 0, (
                f"input_len {input_len} is not a multiple of row dim {ctx.row_dim} "
                "Please check your batch size (power of 2 batch size is recommended)"
            )
            nrows = input_len // ctx.row_dim
            ncols = (ctx.row_dim + 3) // 4 * 4 + 2 * 4
            return nrows * ncols
        elif self._comm_precision == SparseType.MX4:
            assert (
                input_len % 32 == 0
            ), f"input_len {input_len} needs to be multiple of group_size 32"
            # quantized output size = half input size + number of groups (shared exp)
            ctx = none_throws(ctx)
            return (input_len // 2) + (input_len // ctx.mx_group_size)
        else:
            return input_len

    @property
    def quantized_dtype(self) -> torch.dtype:
        return self._comm_precision.as_dtype()

    def create_context(self) -> Optional[QuantizationContext]:
        # fp8 rowwise is activated when row_dim > 0
        if (
            self._comm_precision == SparseType.FP8
            or self._comm_precision == SparseType.MX4
        ):
            return QuantizationContext(self._row_dim)
        # int8 rowwise is default
        return QuantizationContext()
