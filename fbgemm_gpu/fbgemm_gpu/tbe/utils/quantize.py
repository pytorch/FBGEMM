# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[61]

from typing import Optional, Tuple

import torch

from .common import to_device
from fbgemm_gpu.split_embedding_configs import (
    FP8QuantizationConfig,
    SparseType,
)  # usort:skip


def quantize_embs(
    weight: torch.Tensor,
    weight_ty: SparseType,
    fp8_config: Optional[FP8QuantizationConfig] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    weight = weight.detach()
    if weight_ty == SparseType.FP32:
        q_weight = weight.float()
        res_weight = q_weight.view(torch.uint8)
        return (res_weight, None)

    elif weight_ty == SparseType.FP16:
        q_weight = weight.half()
        res_weight = q_weight.view(torch.uint8)
        return (res_weight, None)

    elif weight_ty == SparseType.FP8:
        assert fp8_config is not None
        # Quantize FP32 to HPF8
        res_weight = torch.ops.fbgemm.FloatToHFP8Quantized(
            weight.float(),
            fp8_config.get("exponent_bits"),
            fp8_config.get("exponent_bias"),
            fp8_config.get("max_position"),
        )
        return (res_weight, None)

    elif weight_ty == SparseType.INT8:
        # Note that FloatToFused8BitRowwiseQuantized might have additional padding
        # for alignment if embedding dimension is not a multiple of 4:
        # https://fburl.com/code/z009xsy6
        q_weight = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(weight)
        res_weight = q_weight[:, :-8].view(torch.uint8)
        res_scale_shift = torch.tensor(
            q_weight[:, -8:].view(torch.float32).to(torch.float16).view(torch.uint8)
        )  # [-4, -2]: scale; [-2:]: bias
        return (res_weight, res_scale_shift)

    elif weight_ty == SparseType.INT4 or weight_ty == SparseType.INT2:
        # Note that FP32 -> INT4/INT2 conersion op below might have additional padding
        # for alignment: https://fburl.com/code/xx9kkduf
        q_weight = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
            weight,
            bit_rate=weight_ty.bit_rate(),
        )
        res_weight = q_weight[:, :-4].view(torch.uint8)
        res_scale_shift = torch.tensor(
            q_weight[:, -4:].view(torch.uint8)
        )  # [-4, -2]: scale; [-2:]: bias
        return (res_weight, res_scale_shift)

    else:
        raise RuntimeError("Unsupported SparseType: {}".format(weight_ty))


def dequantize_embs(
    weights: torch.Tensor,
    scale_shift: torch.Tensor,
    weight_ty: SparseType,
    use_cpu: bool,
    fp8_config: Optional[FP8QuantizationConfig] = None,
    # pyre-fixme[7]: Expected `Tensor` but got implicit return value of `None`.
) -> torch.Tensor:
    print(f"weight_ty: {weight_ty}")
    assert (
        weights.dtype == torch.uint8
    ), "The input tensor for dequantize_embs function needs to be byte tensor"
    th_weights = weights

    if scale_shift is not None:
        th_scale_shift: torch.Tensor = scale_shift.view(torch.float16).to(torch.float32)

    if weight_ty == SparseType.INT4:
        (E, D_2) = th_weights.shape
        D = D_2 * 2

        def comp(i: int) -> torch.Tensor:
            subs = th_weights.view(torch.uint8) >> (i * 4)
            sub_mask = subs & 0xF
            result = sub_mask.to(torch.float32) * th_scale_shift[:, 0].reshape(
                -1, 1
            ).to(torch.float32) + th_scale_shift[:, 1].reshape(-1, 1).to(torch.float32)
            return result.to(torch.float32)

        comps = [comp(i) for i in range(2)]
        comps = torch.stack(comps)
        comps = comps.permute(1, 2, 0)
        comps = comps.reshape(E, D)
        return to_device(torch.tensor(comps), use_cpu)

    elif weight_ty == SparseType.INT2:
        (E, D_4) = th_weights.shape
        D = D_4 * 4

        # pyre-fixme[53]: Captured variable `scale_shift` is not annotated.
        # pyre-fixme[53]: Captured variable `weights` is not annotated.
        def comp(i: int) -> torch.Tensor:
            subs = th_weights.view(torch.uint8) >> (i * 2)
            sub_mask = subs & 0x3
            result = sub_mask.to(torch.float32) * th_scale_shift[:, 0].reshape(
                -1, 1
            ).to(torch.float32) + th_scale_shift[:, 1].reshape(-1, 1).to(torch.float32)
            return result.to(torch.float32)

        comps = [comp(i) for i in range(4)]
        comps = torch.stack(comps)
        comps = comps.permute(1, 2, 0)
        comps = comps.reshape(E, D)
        return to_device(torch.tensor(comps), use_cpu)

    elif weight_ty == SparseType.INT8:
        (E, D) = th_weights.shape
        comps = th_weights.to(torch.float32) * th_scale_shift[:, 0].reshape(-1, 1).to(
            torch.float32
        ) + th_scale_shift[:, 1].reshape(-1, 1).to(torch.float32)
        return to_device(torch.tensor(comps), use_cpu)

    elif weight_ty == SparseType.FP8:
        assert fp8_config is not None
        assert scale_shift is None
        # Dequantize HPF8 to FP32
        comps = torch.ops.fbgemm.HFP8QuantizedToFloat(
            weights,
            fp8_config.get("exponent_bits"),
            fp8_config.get("exponent_bias"),
        )
        return to_device(comps, use_cpu)

    elif weight_ty == SparseType.FP16:
        assert scale_shift is None
        comps = th_weights.view(torch.half)
        return to_device(torch.tensor(comps), use_cpu)

    elif weight_ty == SparseType.FP32:
        assert scale_shift is None
        comps = th_weights.view(torch.float32)
        # pyre-fixme[7]: Expected `Tensor` but got implicit return value of `None`.
        return to_device(torch.tensor(comps), use_cpu)


def fake_quantize_embs(
    weights: torch.Tensor,
    scale_shift: Optional[torch.Tensor],
    dequant_weights: torch.Tensor,
    weight_ty: SparseType,
    use_cpu: bool,
    fp8_config: Optional[FP8QuantizationConfig] = None,
) -> None:
    assert (
        weights.dtype == torch.uint8
    ), "The input tensor for dequantize_embs function needs to be byte tensor"
    th_weights = weights

    if scale_shift is not None:
        th_scale_shift: torch.Tensor = (
            scale_shift.contiguous().view(torch.float16).to(torch.float32)
        )

    if weight_ty == SparseType.INT4:
        (E, D_2) = th_weights.shape
        D = D_2 * 2

        def comp(i: int) -> torch.Tensor:
            subs = th_weights.view(torch.uint8) >> (i * 4)
            sub_mask = subs & 0xF
            result = sub_mask.to(torch.float32) * th_scale_shift[:, 0].reshape(
                -1, 1
            ).to(torch.float32) + th_scale_shift[:, 1].reshape(-1, 1).to(torch.float32)
            return result.to(torch.float32)

        comps = [comp(i) for i in range(2)]
        comps = torch.stack(comps)
        comps = comps.permute(1, 2, 0)
        comps = comps.reshape(E, D)
        dequant_weights.copy_(to_device(comps, use_cpu))

    elif weight_ty == SparseType.INT2:
        (E, D_4) = th_weights.shape
        D = D_4 * 4

        # pyre-fixme[53]: Captured variable `scale_shift` is not annotated.
        # pyre-fixme[53]: Captured variable `weights` is not annotated.
        def comp(i: int) -> torch.Tensor:
            subs = th_weights.view(torch.uint8) >> (i * 2)
            sub_mask = subs & 0x3
            result = sub_mask.to(torch.float32) * th_scale_shift[:, 0].reshape(
                -1, 1
            ).to(torch.float32) + th_scale_shift[:, 1].reshape(-1, 1).to(torch.float32)
            return result.to(torch.float32)

        comps = [comp(i) for i in range(4)]
        comps = torch.stack(comps)
        comps = comps.permute(1, 2, 0)
        comps = comps.reshape(E, D)
        dequant_weights.copy_(to_device(comps, use_cpu))

    elif weight_ty == SparseType.INT8:
        (E, D) = th_weights.shape
        comps = th_weights.to(torch.float32) * th_scale_shift[:, 0].reshape(-1, 1).to(
            torch.float32
        ) + th_scale_shift[:, 1].reshape(-1, 1).to(torch.float32)
        dequant_weights.copy_(to_device(comps, use_cpu))

    elif weight_ty == SparseType.FP8:
        assert fp8_config is not None
        assert scale_shift is None
        # Quantize FP32 to HPF8
        comps = torch.ops.fbgemm.FloatToHFP8Quantized(
            dequant_weights.detach().float(),
            fp8_config.get("exponent_bits"),
            fp8_config.get("exponent_bias"),
            fp8_config.get("max_position"),
        )
        weights.copy_(comps)

        # Dequantize HPF8 to FP32
        comps = torch.ops.fbgemm.HFP8QuantizedToFloat(
            comps,
            fp8_config.get("exponent_bits"),
            fp8_config.get("exponent_bias"),
        )
        dequant_weights.copy_(to_device(comps, use_cpu))

    elif weight_ty == SparseType.FP16:
        assert scale_shift is None
        comps = dequant_weights.detach().half().view(torch.uint8)
        weights.copy_(comps)
    elif weight_ty == SparseType.FP32:
        assert scale_shift is None
        comps = dequant_weights.detach().float().view(torch.uint8)
        weights.copy_(comps)
