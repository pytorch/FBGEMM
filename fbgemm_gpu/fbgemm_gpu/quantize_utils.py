#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Optional, Union

import torch

from fbgemm_gpu.triton import dequantize_mx4, quantize_mx4, RoundingMode
from fbgemm_gpu.triton.quantize_ref import py_dequantize_mx4, py_quantize_mx4

logger: logging.Logger = logging.getLogger()

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")

TORCH_HALF_MIN: float = torch.finfo(torch.float16).min
TORCH_HALF_MAX: float = torch.finfo(torch.float16).max

TORCH_BFLOAT16_MIN: float = torch.finfo(torch.bfloat16).min
TORCH_BFLOAT16_MAX: float = torch.finfo(torch.bfloat16).max


def fp32_to_mx4(
    tensor: torch.Tensor,
    group_size: int = 32,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: Optional[Union[RoundingMode, int]] = RoundingMode.ceil,
    stochastic_casting: bool = False,
    use_triton: bool = True,
) -> torch.Tensor:
    """Quantize an FP32 tensor to MX4 with triton or native cuda impl.

    Args:
        tensor (torch.Tensor): FP32 tensor to quantize with M total elements.
        group_size (int): Compute scale in chunks of group_size.
        ebits (int): Number of exponent bits in target mx4 format.
        mbits (int): Number of mantissa bits in target mx4 format.
        rounding_mode (RoundingMode or int): Which type of rounding to use when computing exponent.
        Only supported with use_triton=True.
        stochastic_casting (bool): Whether to use stochastic casting when downcasting.
        use_triton (bool): If set, use triton quantization, otherwise cuda.

    Return:
        output: MX4 tensor packed into int8 values with total elements (M / 2 + M / groupsize)
    """
    # Accelerated MX4 is only available on cuda, if input is on cpu, use python.
    # Operate on flattened input.
    if rounding_mode is None:
        rounding_mode = RoundingMode.ceil

    if not tensor.is_cuda:
        return py_quantize_mx4(
            tensor,
            group_size,
            ebits=ebits,
            mbits=mbits,
            rounding_mode=rounding_mode,
            stochastic_casting=stochastic_casting,
        )

    if use_triton:
        return quantize_mx4(
            tensor,
            group_size,
            ebits=ebits,
            mbits=mbits,
            rounding_mode=rounding_mode,
            stochastic_casting=stochastic_casting,
        )
    else:
        out = torch.ops.fbgemm.quantize_mx_cuda(
            tensor.flatten(),
            scale_bits=8,
            elem_ebits=2,
            elem_mbits=3,
            elem_max_norm=6.0,
            mx_group_size=group_size,
        )
        # Perserve input dimensions.
        output_shape = list(tensor.shape[:-1]) + [-1]
        return out.view(output_shape)


def mx4_to_fp32(
    tensor: torch.Tensor,
    group_size: int = 32,
    use_triton: bool = True,
    ebits: int = 2,
    mbits: int = 1,
) -> torch.Tensor:
    """Dequantize an MX4 tensor to FP32 with triton or native cuda impl.

    Args:
        tensor (torch.Tensor): MX4 packed tensor with total elements (M / 2 + M / groupsize)
        group_size (int): Compute scale in chunks of group_size.
        use_triton (bool): If set, use triton quantization, otherwise cuda.
        ebits (int): Number of exponent bits in target mx4 format.
        mbits (int): Number of mantissa bits in target mx4 format.

    Return:
        output: FP32 tensor with total elements (M).
    """
    # Accelerated MX4 dequantize is only available on cuda, if input is on cpu, use python.
    if not tensor.is_cuda:
        return py_dequantize_mx4(tensor, group_size, ebits=ebits, mbits=mbits)
    if use_triton:
        return dequantize_mx4(tensor, group_size, ebits=ebits, mbits=mbits)
    else:
        return torch.ops.fbgemm.dequantize_mx_cuda(tensor.flatten(), group_size)


def fp32_to_fp16_with_clamp(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor, TORCH_HALF_MIN, TORCH_HALF_MAX).half()


def fp32_to_bf16_with_clamp(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor, TORCH_BFLOAT16_MIN, TORCH_BFLOAT16_MAX).bfloat16()


def fp32_to_hfp8_with_clamp(
    tensor: torch.Tensor, ebits: int = 4, mbits: int = 3, bias: int = 15
) -> torch.Tensor:
    max_pos: float = (2 ** ((1 << ebits) - 2 - bias)) * (2 - 2 ** (-mbits))
    return torch.ops.fbgemm.FloatToHFP8Quantized(
        tensor.contiguous(),
        ebits,
        bias,
        max_pos,
    )


def fp16_to_fp32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float()


def bf16_to_fp32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.bfloat16).float()


def hfp8_to_fp32(tensor: torch.Tensor, ebits: int = 4, bias: int = 15) -> torch.Tensor:
    return torch.ops.fbgemm.HFP8QuantizedToFloat(
        tensor.contiguous().view(torch.uint8),
        ebits,
        bias,
    )


def measure_fp16_quant_error(input_tensor: torch.Tensor) -> None:
    # TODO: log to tensorboard

    num_nan_fp32_tensor = torch.numel(input_tensor[torch.isnan(input_tensor)])
    logger.info(
        "num NaN in fp32 tensor: {}, ratio: {}.".format(
            num_nan_fp32_tensor, num_nan_fp32_tensor / torch.numel(input_tensor)
        )
    )

    logger.info(
        "fp32 tensor profile: min: {}, max: {}, min abs:{}, max abs:{}.".format(
            torch.min(input_tensor),
            torch.max(input_tensor),
            torch.min(torch.abs(input_tensor)),
            torch.max(torch.abs(input_tensor)),
        )
    )

    fp16_tensor = fp32_to_fp16_with_clamp(input_tensor)
    num_nan_fp16_tensor = torch.numel(fp16_tensor[torch.isnan(fp16_tensor)])

    logger.info(
        "num NaN in fp16 tensor: {}, ratio: {}.".format(
            num_nan_fp16_tensor, num_nan_fp16_tensor / torch.numel(input_tensor)
        )
    )

    diff = torch.abs(input_tensor - fp16_tensor.float())
    rel_diff = diff / torch.abs(input_tensor)
    logger.info(
        "fp32_to_fp16 abs error: min={}, max={}, avg={}.".format(
            torch.min(diff), torch.max(diff), torch.mean(diff)
        )
    )

    rel_diff_not_nan = rel_diff[torch.logical_not(torch.isnan(rel_diff))]
    logger.info(
        "fp32_to_fp16 rel error: min={}, max={}, avg={}.".format(
            torch.min(rel_diff_not_nan),
            torch.max(rel_diff_not_nan),
            torch.mean(rel_diff_not_nan),
        )
    )

    rel_diff_1_idx = torch.where(rel_diff == 1.0)
    fp32_rel_err_1_vals = input_tensor[rel_diff_1_idx]
    if torch.numel(fp32_rel_err_1_vals) > 0:
        fp32_rel_err_1_vals = torch.abs(fp32_rel_err_1_vals)
        logger.info(
            "fp32_to_fp16 rel error == 1: fp32 min:{}, fp32 max:{}, fp32 avg:{}.".format(
                torch.min(fp32_rel_err_1_vals),
                torch.max(fp32_rel_err_1_vals),
                torch.mean(fp32_rel_err_1_vals),
            )
        )

        subrange_ratio = torch.numel(fp16_tensor[rel_diff_1_idx]) / torch.numel(
            fp16_tensor
        )
        logger.info("sub fp16 range ratio: {}".format(subrange_ratio))
