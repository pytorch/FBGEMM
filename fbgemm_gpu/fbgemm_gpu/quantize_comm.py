#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Type

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

logger: logging.Logger = logging.getLogger()
from abc import ABC, abstractproperty


# The code in this file is refactored from https://fburl.com/code/p2gy2gxb

# FP8 configurations
ebits, mbits, bias = 4, 3, 15
max_pos: float = (2 ** ((1 << ebits) - 2 - bias)) * (2 - 2 ** (-mbits))


class QuantizeCommCodecIf(ABC):
    @abstractproperty
    def encoder(self) -> Type[torch.autograd.Function]:
        ...

    @abstractproperty
    def decoder(self) -> Type[torch.autograd.Function]:
        ...


class QuantizeWork:
    def __init__(
        self,
        comm_precision: SparseType,
    ) -> None:
        self.comm_precision = comm_precision
        logger.info(f"Creating QuantizeWork comm_precision:{comm_precision}")

    def quantize_tensor(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if self.comm_precision == SparseType.FP32:
            return input_tensor
        elif self.comm_precision == SparseType.FP16:
            return fp32_to_fp16_with_clamp(input_tensor)
        elif self.comm_precision == SparseType.BF16:
            # View BF16 tensor as half tensor to bypass autograd restriction
            return fp32_to_bf16_with_clamp(input_tensor).view(torch.float16)
        elif self.comm_precision == SparseType.FP8:
            # View FP8 tensor as half tensor to bypass autograd restriction
            return fp32_to_hfp8_with_clamp(input_tensor, ebits, mbits, bias).view(
                torch.float16
            )
        else:
            raise ValueError(f"comm_precision={self.comm_precision} is not supported")

    def dequantize_tensor(
        self,
        quantized_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if self.comm_precision == SparseType.FP32:
            return quantized_tensor
        elif self.comm_precision == SparseType.FP16:
            return fp16_to_fp32(quantized_tensor)
        elif self.comm_precision == SparseType.BF16:
            return bf16_to_fp32(quantized_tensor)
        elif self.comm_precision == SparseType.FP8:
            return hfp8_to_fp32(quantized_tensor, ebits, bias)
        else:
            raise ValueError(f"comm_precision={self.comm_precision} is not supported")


class QuantizeCommCodec(QuantizeCommCodecIf):
    def __init__(
        self,
        fwd_comm_precision: SparseType,
        bwd_comm_precision: SparseType,
        loss_scale: Optional[float] = None,
    ) -> None:

        logger.info(
            f"Creating QuantizeCommCodec fwd_comm_precision:{fwd_comm_precision} bwd_comm_precision:{bwd_comm_precision} "
        )
        if loss_scale is not None:
            if bwd_comm_precision not in [SparseType.FP16]:
                logger.warning(
                    f"Setting loss scale for bwd_comm_precision={bwd_comm_precision} is not supported. Overriding to None"
                )
                loss_scale = None

        fwd_work = QuantizeWork(
            comm_precision=fwd_comm_precision,
        )
        bwd_work = QuantizeWork(
            comm_precision=bwd_comm_precision,
        )

        class Encoder(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                return fwd_work.quantize_tensor(input_tensor)

            @staticmethod
            def backward(ctx, grad_output):
                if loss_scale is not None:
                    grad_output = grad_output / loss_scale
                return bwd_work.dequantize_tensor(grad_output)

        class Decoder(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                return fwd_work.dequantize_tensor(input_tensor)

            @staticmethod
            def backward(ctx, grad_output):
                if loss_scale is not None:
                    grad_output = grad_output * loss_scale
                return bwd_work.quantize_tensor(grad_output)

        self._encoder = Encoder
        self._decoder = Decoder

    @property
    def encoder(self) -> Type[torch.autograd.Function]:
        return self._encoder

    @property
    def decoder(self) -> Type[torch.autograd.Function]:
        return self._decoder
