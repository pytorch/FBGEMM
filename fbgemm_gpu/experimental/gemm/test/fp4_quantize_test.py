# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest
from typing import Optional

import fbgemm_gpu
import torch
from fbgemm_gpu.experimental.gemm.triton_gemm.fp4_quantize import (
    _to_blocked,
    triton_quantize_mx4_unpack,
    triton_rms_quantize_mx4_unpack,
    triton_scale_nvfp4_quant,
    triton_scale_nvfp4_quant_rms,
    triton_scale_nvfp4_quant_silu,
    triton_silu_quantize_mx4_unpack,
)
from fbgemm_gpu.quantize_utils import fp32_to_mx4, RoundingMode
from torch import Tensor

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if not open_source:
    from gen_ai.llm_inference.fb.llm.kernel.rms_norm import rms_norm
    from gen_ai.llm_inference.fb.llm.kernel.silu_mul import silu_mul


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestFp4Quantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_quantize_fp4(self) -> None:
        def _test_quantize_fp4(
            shape: tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 32
            rounding_mode = RoundingMode.even
            packed_group_size = group_size // 2
            groups_per_row = math.ceil(N / group_size)
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            xq_ref, x_scale_ref = triton_quantize_mx4_unpack(
                x, group_size=group_size, rounding_mode=rounding_mode
            )
            xq_packed = fp32_to_mx4(
                x, group_size=group_size, rounding_mode=rounding_mode
            )

            xq = torch.empty([M, N // 2], device=x.device, dtype=torch.uint8)
            x_scale = torch.empty(
                [M, groups_per_row], device=x.device, dtype=torch.uint8
            )

            for i in range(groups_per_row):
                start_idx = i * (packed_group_size + 1)
                end_idx = start_idx + packed_group_size
                xq[:, i * packed_group_size : (i + 1) * packed_group_size] = xq_packed[
                    :, start_idx:end_idx
                ]
                x_scale[:, i] = xq_packed[:, end_idx]

            self.assertTrue(torch.equal(xq, xq_ref))
            self.assertTrue(
                torch.equal(_to_blocked(x_scale), x_scale_ref.view(torch.uint8))
            )

        _test_quantize_fp4((1, 128))
        _test_quantize_fp4((3, 512))
        _test_quantize_fp4((128, 1024))
        _test_quantize_fp4((4096, 10240))


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestFp4RmsQuantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_rms_quantize_fp4(self) -> None:
        def _test_rms_quantize_fp4(
            shape: tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 32
            rounding_mode = RoundingMode.even
            packed_group_size = group_size // 2
            groups_per_row = math.ceil(N / group_size)
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            w = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            xq_ref, x_scale_ref = triton_rms_quantize_mx4_unpack(
                x, w, EPS=1e-5, group_size=group_size, rounding_mode=rounding_mode
            )

            intermediate = (
                x.to(torch.float32).reshape(-1, group_size)
                * torch.rsqrt(
                    torch.pow(x.to(torch.float32).reshape(-1, group_size), 2).mean(
                        dim=1
                    )
                    + 1e-5
                ).unsqueeze(1)
            ) * w.reshape(-1, group_size).to(torch.float32)

            intermediate = intermediate.to(torch.bfloat16).reshape(M, N)
            xq_packed = fp32_to_mx4(
                intermediate, group_size=group_size, rounding_mode=rounding_mode
            )

            xq = torch.empty([M, N // 2], device=x.device, dtype=torch.uint8)
            x_scale = torch.empty(
                [M, groups_per_row], device=x.device, dtype=torch.uint8
            )

            for i in range(groups_per_row):
                start_idx = i * (packed_group_size + 1)
                end_idx = start_idx + packed_group_size
                xq[:, i * packed_group_size : (i + 1) * packed_group_size] = xq_packed[
                    :, start_idx:end_idx
                ]
                x_scale[:, i] = xq_packed[:, end_idx]

            self.assertTrue(torch.equal(xq, xq_ref))
            self.assertTrue(
                torch.equal(_to_blocked(x_scale), x_scale_ref.view(torch.uint8))
            )

        _test_rms_quantize_fp4((1, 32))
        _test_rms_quantize_fp4((1, 128))
        _test_rms_quantize_fp4((3, 512))
        _test_rms_quantize_fp4((128, 1024))
        # TODO: fix potential bug with large tensors
        # _test_rms_quantize_fp4((4096, 10240))


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestFp4SiluQuantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_silu_quantize_fp4(self) -> None:
        def _test_silu_quantize_fp4(
            shape: tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 32
            rounding_mode = RoundingMode.even
            packed_group_size = group_size // 2
            groups_per_row = math.ceil(N / group_size)
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            w = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            xq_ref, x_scale_ref = triton_silu_quantize_mx4_unpack(
                x, w, group_size=group_size, rounding_mode=rounding_mode
            )
            intermediate = torch.nn.functional.silu(x.to(torch.float32)) * w.to(
                torch.float32
            )
            intermediate = intermediate.to(torch.bfloat16)
            xq_packed = fp32_to_mx4(
                intermediate, group_size=group_size, rounding_mode=rounding_mode
            )

            xq = torch.empty([M, N // 2], device=x.device, dtype=torch.uint8)
            x_scale = torch.empty(
                [M, groups_per_row], device=x.device, dtype=torch.uint8
            )

            for i in range(groups_per_row):
                start_idx = i * (packed_group_size + 1)
                end_idx = start_idx + packed_group_size
                xq[:, i * packed_group_size : (i + 1) * packed_group_size] = xq_packed[
                    :, start_idx:end_idx
                ]
                x_scale[:, i] = xq_packed[:, end_idx]

            self.assertTrue(torch.equal(xq, xq_ref))
            self.assertTrue(
                torch.equal(_to_blocked(x_scale), x_scale_ref.view(torch.uint8))
            )

        _test_silu_quantize_fp4((1, 128))
        _test_silu_quantize_fp4((3, 512))
        _test_silu_quantize_fp4((128, 1024))
        _test_silu_quantize_fp4((10240, 10240))


def _n_ones(n: int) -> int:
    return (1 << n) - 1


EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)


# copy-pasted from https://github.com/pytorch/ao/blob/4f5bc7a137eff86d1348a1c78287f5a76bf7e10a/torchao/prototype/custom_fp_utils.py#L27
# TODO once the reference implementation is landed into PyTorch, use it instead
def _f32_to_floatx_unpacked(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers with the given
    number of exponent and mantissa bits.

    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding

    Note: there are no special values (NaN, inf) support in this code. Values
    outside the representable range of Floatx after rounding are clamped to the
    maximum Floatx magnitude (sign is preserved).

    Code below is an adaptation of https://fburl.com/code/ciwofcg4

    Background 1: last answer in https://stackoverflow.com/questions/8981913/how-to-perform-round-to-even-with-floating-point-numbers  # noqa: E501
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    # TODO document this better
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # all E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign

    # TODO: can the branch floating point comparisons below be done without
    # converting to float? probably but need to verify
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    #
    # branch 1: saturate to max val - handled later in the code which combines
    #   the branches
    #

    #
    # branch 2: to conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1


def f32_to_f4_unpacked(x):
    """
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, with bits 0-3 empty and
      bits 4-7 in fp4_e2m1
    """
    return _f32_to_floatx_unpacked(x, EBITS_F4_E2M1, MBITS_F4_E2M1)


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


# copy-pasted from https://github.com/pytorch/ao/blob/4f5bc7a137eff86d1348a1c78287f5a76bf7e10a/torchao/prototype/mx_formats/kernels.py#L756
# TODO once the reference implementation is landed into PyTorch, use it instead
def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[::2] | uint8_data[1::2] << 4).view(down_size(shape))


F4_E2M1_MAX = 6.0
E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


# copy-pasted from https://github.com/pytorch/ao/blob/4f5bc7a137eff86d1348a1c78287f5a76bf7e10a/torchao/prototype/mx_formats/nvfp4_tensor.py#L676
# TODO once the reference implementation is landed into PyTorch, use it instead
def nvfp4_quantize(
    data_hp: torch.Tensor,
    block_size: int = 16,
    per_tensor_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVIDIA FP4 quantization with UE4M3 scales.

    Implements the NVIDIA algorithm for quantizing tensors to FP4 format
    with unsigned E4M3 (UE4M3) scales.

    Args:
        data_hp: High precision input tensor (bfloat16 or float32)
        block_size: Block size for quantization (must be 16)
        per_tensor_amax: Optional pre-computed absolute maximum for calibration.
            If provided, uses per-tensor scaling. If None, uses block-wise scaling only.

    Returns:
        tuple: A tuple containing:
            - total_scale_fp8: Blockwise scales in float8_e4m3fn format
            - per_tensor_scale: Global per-tensor scale if per_tensor_amax provided, else None
            - data_lp: Packed FP4 data (2 values per byte)

    Raises:
        AssertionError: If input dtype is not supported, tensor size is not
            divisible by block_size, tensor is not contiguous, or block_size != 16
    """
    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} not supported"
    assert data_hp.size(-1) % block_size == 0, "K dim must be divisible by block_size"
    assert data_hp.is_contiguous(), "Only support contiguous data for now"
    assert block_size == 16, "NVFP4 requires block_size=16"

    orig_shape = data_hp.shape
    # Convert to float32 early for consistent precision with Triton implementation
    data_hp = data_hp.float().reshape(orig_shape[0], -1, block_size)

    max_abs = torch.amax(torch.abs(data_hp), dim=-1)
    # These scales are currently in fp32, we are going to `quantize` them to e4m3
    block_scale = max_abs / F4_E2M1_MAX

    out_scales = None
    if per_tensor_scale is None:
        # We are doing single level scaling
        block_scale_fp8 = torch.clamp(block_scale, min=E4M3_EPS, max=F8E4M3_MAX).to(
            torch.float8_e4m3fn
        )
        block_scale_fp32 = block_scale_fp8.to(torch.float32)
        data_scaled = data_hp / block_scale_fp32.unsqueeze(-1)
        out_scales = block_scale_fp8
    else:
        # We are doing two level scaling,
        # This will likely be calibrated but
        # we want the per_tensor_scale ~= amax of the block_scale_fp32
        block_scale_fp32 = block_scale.to(torch.float32)
        # Quantize the blockwise scales w/ the per_tensor_scale
        scaled_block_scales = block_scale_fp32 / per_tensor_scale
        scaled_block_scales_fp8 = torch.clamp(
            scaled_block_scales, min=E4M3_EPS, max=F8E4M3_MAX
        ).to(torch.float8_e4m3fn)
        scaled_block_scales_fp32 = scaled_block_scales_fp8.to(torch.float32)
        # We "temporarily" dequant the scaled_block_scales_fp32 to get the per_tensor_scale
        # To apply to data
        total_scale = per_tensor_scale * scaled_block_scales_fp32
        data_scaled = data_hp / total_scale.unsqueeze(-1)
        out_scales = scaled_block_scales_fp8

    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_scaled = data_scaled.view(orig_shape)
    data_lp = f32_to_f4_unpacked(data_scaled)
    # TODO: NotImplementedError: "copy_kernel" not implemented for 'Float4_e2m1fn_x2'
    # data_lp = pack_uint4(data_lp).view(torch.float4_e2m1fn_x2)
    data_lp = pack_uint4(data_lp)
    return out_scales, data_lp


def per_tensor_amax_to_scale(amax: torch.Tensor) -> torch.Tensor:
    """Convert per-tensor amax to per-tensor scale for NVFP4 quantization.

    Divides by both F8E4M3_MAX and F4_E2M1_MAX to ensure block scales can utilize
    the full FP8 E4M3 range (up to 448) when block_max equals tensor_max.
    Without F4_E2M1_MAX, the maximum scale would only reach FP8_MAX / FP4_MAX.

    Args:
        amax: Per-tensor absolute maximum value from calibration

    Returns:
        torch.Tensor: Per-tensor scale for two-level NVFP4 scaling
    """
    return amax.to(torch.float32) / (F8E4M3_MAX * F4_E2M1_MAX)


def ceil_div(a, b):
    return (a + b - 1) // b


# copy-pasta from https://github.com/pytorch/ao/blob/4f5bc7a137eff86d1348a1c78287f5a76bf7e10a/torchao/prototype/mx_formats/utils.py#L32
# TODO once the reference implementation is landed into PyTorch, use it instead
def to_blocked(input_matrix) -> Tensor:
    """
    Rearrange a large matrix by breaking it into blocks and applying the rearrangement pattern.

    See:
        https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)
        use_triton_kernel: Whether to use a triton implementation instead of relying on
            torch.compile

    Returns:
        Rearranged tensor of shape (32*ceil_div(H,128), 16*ceil_div(W,4))
    """

    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    # TODO This is to work around VLLM's usage of compile w/ dynamic shapes
    if torch.compiler.is_compiling() or (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestNVFp4Quantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_quantize_nvfp4(self) -> None:

        def _test_quantize_nvfp4(
            shape: tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 16
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            x_global_scale = torch.tensor([448.0 * 6.0]).to(
                device=x.device
            ) / torch.amax(x.flatten(), dim=-1)
            x_global_scale = torch.tensor(1.0, device=x.device)
            x_global_scale = per_tensor_amax_to_scale(torch.amax(torch.abs(x)))

            x_scale_ref, xq_ref = nvfp4_quantize(
                x,
                group_size,
                x_global_scale,
            )
            x_scale_ref = to_blocked(x_scale_ref)
            xq, x_scale = triton_scale_nvfp4_quant(
                x,
                x_global_scale.reciprocal(),
                group_size=group_size,
            )
            x_scale = x_scale.view(torch.float8_e4m3fn).view(*x_scale_ref.shape)

            torch.testing.assert_close(xq, xq_ref, atol=0, rtol=0)
            torch.testing.assert_close(x_scale, x_scale_ref, atol=0, rtol=0)

        # TODO(future PR): fix the kernel to enable the following two tests
        # the issue: today the scale tensor is initialized with torch.empty, and
        # the triton kernel does not properly set the padded scale elements to
        # zero.

        # _test_quantize_nvfp4((1, 128))
        # _test_quantize_nvfp4((4, 512))

        _test_quantize_nvfp4((128, 1024))
        _test_quantize_nvfp4((10240, 10240))


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestNVFp4SiluQuantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    @unittest.skipIf(open_source, "silu_mul is not available")
    def test_silu_quantize_nvfp4(self) -> None:

        def _test_silu_quantize_nvfp4(
            shape: tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 16
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            w = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            x_global_scale = torch.tensor([448.0 * 6.0]).to(
                device=x.device
            ) / torch.amax(x.flatten(), dim=-1)
            xq_ref, x_scale_ref = triton_scale_nvfp4_quant_silu(
                x,
                w,
                x_global_scale,
                group_size=group_size,
            )

            intermediate = silu_mul(x.reshape(-1, 16), w.reshape(-1, 16))
            intermediate = intermediate.to(torch.bfloat16).reshape(M, N)
            xq, x_scale = triton_scale_nvfp4_quant(
                intermediate,
                x_global_scale,
                group_size=group_size,
            )

            self.assertTrue(torch.equal(xq, xq_ref))
            self.assertTrue(torch.equal(x_scale, x_scale_ref))

        _test_silu_quantize_nvfp4((1, 128))
        _test_silu_quantize_nvfp4((4, 512))
        _test_silu_quantize_nvfp4((128, 1024))
        _test_silu_quantize_nvfp4((10240, 10240))


class TestNVFp4RmsQuantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    @unittest.skipIf(open_source, "rms_norm is not available")
    def test_rms_quantize_nvfp4(self) -> None:

        def _test_rms_quantize_nvfp4(
            shape: tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 16
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            w = torch.randn(group_size, dtype=torch.bfloat16, device=device)
            x_global_scale = torch.tensor([448.0 * 6.0]).to(
                device=x.device
            ) / torch.amax(x.flatten(), dim=-1)
            xq_ref, x_scale_ref = triton_scale_nvfp4_quant_rms(
                x,
                w.repeat(M * N // group_size),
                x_global_scale,
                group_size=group_size,
                EPS=1e-5,
            )

            intermediate = rms_norm(x.reshape(-1, 16), w, eps=1e-5)
            intermediate = intermediate.to(torch.bfloat16).reshape(M, N)
            xq, x_scale = triton_scale_nvfp4_quant(
                intermediate,
                x_global_scale,
                group_size=group_size,
            )

            self.assertTrue(torch.equal(xq, xq_ref))
            self.assertTrue(torch.equal(x_scale, x_scale_ref))

        _test_rms_quantize_nvfp4((1, 128))
        _test_rms_quantize_nvfp4((4, 512))
        _test_rms_quantize_nvfp4((128, 1024))
        _test_rms_quantize_nvfp4((1024, 10240))
        # Note, large testing tensors may lead to slight numerical differences
