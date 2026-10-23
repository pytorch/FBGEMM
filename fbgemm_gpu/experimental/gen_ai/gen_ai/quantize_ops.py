# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch


def _native_fp8_dtypes(input: torch.Tensor) -> tuple[torch.dtype, torch.dtype]:
    if torch.version.hip:
        device = input.device
        if device.type == "cuda" or (
            device.type == "cpu" and torch.cuda.is_available()
        ):
            index = (
                device.index if device.type == "cuda" else torch.cuda.current_device()
            )
            arch = torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]
            if arch.startswith(("gfx94", "gfx90a")):
                return torch.float8_e4m3fnuz, torch.float8_e5m2fnuz
    return torch.float8_e4m3fn, torch.float8_e5m2


def _fake_output_device(input: torch.Tensor) -> torch.device:
    if input.device.type == "cpu":
        return torch.accelerator.current_accelerator() or input.device
    return input.device


def _quantized_meta(
    input: torch.Tensor,
    output_dtype: Optional[torch.dtype],
    allow_e5m2: bool = False,
    output_device: Optional[torch.device] = None,
) -> torch.Tensor:
    torch._check(input.dim() >= 2, lambda: "The dim of input should be at least 2")
    e4m3, e5m2 = _native_fp8_dtypes(input)
    dtype = output_dtype if output_dtype is not None else e4m3
    if input.device.type == "meta":
        allowed = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        if allow_e5m2:
            allowed += (torch.float8_e5m2, torch.float8_e5m2fnuz)
    else:
        allowed = (e4m3, e5m2) if allow_e5m2 else (e4m3,)
    torch._check(
        dtype in allowed, lambda: "output_dtype must match the device native FP8 dtype"
    )
    return torch.empty_like(
        input,
        dtype=dtype,
        device=output_device if output_device is not None else input.device,
    )


@torch.library.register_fake("fbgemm::quantize_fp8_per_tensor", allow_override=True)
def quantize_fp8_per_tensor(
    input: torch.Tensor,
    bs: Optional[torch.Tensor] = None,
    scale_ub: Optional[torch.Tensor] = None,
    stochastic_rounding: bool = False,
    output_dtype: Optional[torch.dtype] = None,
) -> list[torch.Tensor]:
    device = _fake_output_device(input)
    return [
        _quantized_meta(input, output_dtype, output_device=device),
        input.new_empty((), dtype=torch.float32, device=device),
    ]


@torch.library.register_fake("fbgemm::quantize_fp8_per_row", allow_override=True)
def quantize_fp8_per_row(
    input: torch.Tensor,
    bs: Optional[torch.Tensor] = None,
    scale_ub: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
    stochastic_rounding: bool = False,
) -> list[torch.Tensor]:
    # The eager rowwise operator copies CPU-input results back to CPU.
    torch._check(
        output_dtype is None
        or output_dtype
        in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        ),
        lambda: "output must be e4m3 or e5m2 (output_dtype)",
    )
    quantized = _quantized_meta(input, output_dtype, allow_e5m2=True)
    if stochastic_rounding:
        torch._check(
            quantized.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz),
            lambda: "Stochastic rounding is only supported for FP8 E4M3 output",
        )
    return [
        quantized,
        input.new_empty((math.prod(input.shape[:-1]),), dtype=torch.float32),
    ]


@torch.library.register_fake("fbgemm::quantize_fp8_per_col", allow_override=True)
def quantize_fp8_per_col(
    input: torch.Tensor,
    bs: Optional[torch.Tensor] = None,
    scale_ub: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> list[torch.Tensor]:
    device = _fake_output_device(input)
    return [
        _quantized_meta(input, output_dtype, output_device=device),
        input.new_empty(input.shape[1:], dtype=torch.float32, device=device),
    ]
