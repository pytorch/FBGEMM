#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


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
