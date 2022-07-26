#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Optional

import torch

from fbgemm_gpu.quantize_utils import (
    bf16_to_fp32,
    fp16_to_fp32,
    fp32_to_bf16_with_clamp,
    fp32_to_fp16_with_clamp,
    fp32_to_hfp8_with_clamp,
    hfp8_to_fp32,
)
from fbgemm_gpu.split_embedding_configs import SparseType
from torch.autograd.profiler import record_function

logger: logging.Logger = logging.getLogger()


def _quantize_tensor(
    input_tensor: torch.Tensor,
    comm_precision: SparseType,
) -> torch.Tensor:
    if comm_precision == SparseType.FP32:
        return input_tensor
    elif comm_precision == SparseType.FP16:
        return fp32_to_fp16_with_clamp(input_tensor)
    elif comm_precision == SparseType.BF16:
        return fp32_to_bf16_with_clamp(input_tensor)
    elif comm_precision == SparseType.FP8:
        return fp32_to_hfp8_with_clamp(input_tensor)
    else:
        raise ValueError(f"comm_precision={comm_precision} is not supported")


def _dequantize_tensor(
    quantized_tensor: torch.Tensor,
    comm_precision: SparseType,
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
        assert quantized_tensor.dtype == torch.uint8
        return hfp8_to_fp32(quantized_tensor)
    else:
        raise ValueError(f"comm_precision={comm_precision} is not supported")


class QuantizedCommCodec:
    def __init__(
        self,
        comm_precision: SparseType,
        loss_scale: Optional[float] = None,
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

    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self._loss_scale is not None:
            input_tensor = self._loss_scale * input_tensor
        with record_function(
            f"## encoder {self._comm_precision} {self._loss_scale} ##"
        ):
            return _quantize_tensor(input_tensor, self._comm_precision)

    def decode(self, input_grad: torch.Tensor) -> torch.Tensor:
        if self._loss_scale is not None:
            input_grad = input_grad / self._loss_scale
        with record_function(
            f"## decoder {self._comm_precision} {self._loss_scale} ##"
        ):
            dequantized_tensor = _dequantize_tensor(input_grad, self._comm_precision)
        return dequantized_tensor

    @property
    def quantized_dtype(self) -> torch.dtype:
        if self._comm_precision == SparseType.FP16:
            return torch.half
        elif self._comm_precision == SparseType.BF16:
            return torch.bfloat16
        elif self._comm_precision == SparseType.FP8:
            return torch.uint8
        return torch.float
