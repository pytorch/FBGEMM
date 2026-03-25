# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Accuracy testing philosophy:
#
# fp16 matmul implementations (cublass/cutlass, Triton, TLX WGMMA) all accumulate
# K products in fp32 but with different tiling and reduction orders. This
# means two correct fp16 kernels can produce results that differ at the
# last few bits of fp16 precision — comparing them directly with tight fp16
# tolerances produces false failures that are purely accumulation-order
# artifacts, not kernel bugs.
#
# Instead we measure error against a fp32 ground truth (same inputs promoted
# to fp32 before matmul). cuBLAS fp16's error vs fp32 establishes the
# baseline — any correct fp16 kernel should have error in the same ballpark.
# We assert the kernel's error is within a small multiplier of this baseline.

import sys

import pytest
import torch
from ikbo.ops.tlx_ikbo_lce import create_user_flag, tlx_ikbo_lce
from ikbo.ops.torch_lce import torch_decomposed_lce
from ikbo.ops.triton_ikbo_lce import triton_ikbo_lce

DEVICE = "cuda"
DTYPE = torch.float16
PAD_UNIT = 8

# Allow kernel error up to a certain range of fp16 baseline error.
# Small absolute floor handles cases where cuBLAS is near-exact.
ERROR_MULTIPLIER = 1.0
ERROR_FLOOR = 1e-4


def _prepare_inputs(B, M, N, K_USER, K_CAND, cand_to_user_ratio):
    """Create padded test inputs with a fixed candidate-to-user ratio."""
    k_user = ((K_USER + PAD_UNIT - 1) // PAD_UNIT) * PAD_UNIT
    k_cand = ((K_CAND + PAD_UNIT - 1) // PAD_UNIT) * PAD_UNIT

    num_users = (B + cand_to_user_ratio - 1) // cand_to_user_ratio
    cand_to_user_index = (torch.arange(B, device=DEVICE) // cand_to_user_ratio).int()

    cw_cand = torch.randn((M, k_cand), device=DEVICE, dtype=DTYPE)
    cw_user = torch.randn((M, k_user), device=DEVICE, dtype=DTYPE)
    e_cand = torch.randn((B, k_cand, N), device=DEVICE, dtype=DTYPE)
    e_user = torch.randn((num_users, k_user, N), device=DEVICE, dtype=DTYPE)

    return cw_cand, cw_user, e_cand, e_user, cand_to_user_index


def _check_vs_fp32(name, out, ref_fp16, ref_fp32):
    """Assert kernel error vs fp32 is comparable to cuBLAS fp16 error vs fp32.

    Measures both cuBLAS fp16 and the kernel against the fp32 ground truth,
    then asserts the kernel's max error does not significantly exceed the
    cuBLAS baseline. This avoids false failures from accumulation-order
    differences between two correct fp16 implementations.
    """
    baseline_err = (ref_fp16.float() - ref_fp32).abs().max().item()
    kernel_err = (out.float() - ref_fp32).abs().max().item()
    threshold = max(ERROR_MULTIPLIER * baseline_err, ERROR_FLOOR)
    ratio = kernel_err / baseline_err if baseline_err > 0 else float("inf")

    assert kernel_err <= threshold, (
        f"{name} error exceeds {ERROR_MULTIPLIER}x cuBLAS fp16 baseline: "
        f"kernel={kernel_err:.4e}, baseline={baseline_err:.4e}, ratio={ratio:.2f}x"
    )


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Ensure reproducibility for every test."""
    torch.manual_seed(0)


@pytest.mark.parametrize("B", [512, 1024])
@pytest.mark.parametrize("M", [128, 433])
@pytest.mark.parametrize("N", [256])
@pytest.mark.parametrize("K_USER", [1024, 1025])
@pytest.mark.parametrize("K_CAND", [1024, 1023])
@pytest.mark.parametrize("cand_to_user_ratio", [10, 70])
def test_triton_ikbo_lce(B, M, N, K_USER, K_CAND, cand_to_user_ratio):
    cw_c, cw_u, e_c, e_u, idx = _prepare_inputs(
        B, M, N, K_USER, K_CAND, cand_to_user_ratio
    )

    ref_fp32 = torch_decomposed_lce(
        cw_c.float(), cw_u.float(), e_c.float(), e_u.float(), idx
    )
    ref_fp16 = torch_decomposed_lce(cw_c, cw_u, e_c, e_u, idx)
    out = triton_ikbo_lce(cw_c, cw_u, e_c, e_u, idx)

    _check_vs_fp32("triton", out, ref_fp16, ref_fp32)


@pytest.mark.parametrize("B", [512, 1024])
@pytest.mark.parametrize("M", [128, 433])
@pytest.mark.parametrize("N", [256])
@pytest.mark.parametrize("K_USER", [1024, 1184])
@pytest.mark.parametrize("K_CAND", [1024, 872])
@pytest.mark.parametrize("cand_to_user_ratio", [100, 1000])
def test_tlx_ikbo_lce(B, M, N, K_USER, K_CAND, cand_to_user_ratio):
    cw_c, cw_u, e_c, e_u, idx = _prepare_inputs(
        B, M, N, K_USER, K_CAND, cand_to_user_ratio
    )

    ref_fp32 = torch_decomposed_lce(
        cw_c.float(), cw_u.float(), e_c.float(), e_u.float(), idx
    )
    ref_fp16 = torch_decomposed_lce(cw_c, cw_u, e_c, e_u, idx)

    user_flag = create_user_flag(cw_u, e_u)
    out = tlx_ikbo_lce(cw_c, cw_u, e_c, e_u, idx, user_flag)

    _check_vs_fp32("tlx", out, ref_fp16, ref_fp32)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
