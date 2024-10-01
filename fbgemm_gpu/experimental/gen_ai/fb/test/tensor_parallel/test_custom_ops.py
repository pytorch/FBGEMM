# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest
from typing import List

import fbgemm_gpu.experimental.gen_ai  # noqa: F401
import hypothesis.strategies as st

import torch
from hypothesis import given, settings, Verbosity

VERBOSITY: Verbosity = Verbosity.verbose


def _get_tensor(
    shape: List[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    T = torch.empty(*shape, dtype=dtype, device=device)
    if dtype != torch.float8_e4m3fn:
        T = torch.rand(*shape, dtype=dtype, device=device) - 0.5
    else:
        T_uint8 = torch.randint(1, 64, shape, dtype=torch.uint8, device=device)
        T.copy_(T_uint8)
    return T


def _check_tensor_all_close(
    T: torch.Tensor, T_ref: torch.Tensor, atol=None, rtol=None
) -> None:
    assert T.dtype == T_ref.dtype
    if T.dtype == torch.float8_e4m3fn:
        T_uint8 = torch.empty_like(T, dtype=torch.uint8, device=T.device)
        T_uint8.copy_(T)
        T_ref_uint8 = torch.empty_like(T_ref, dtype=torch.uint8, device=T.device)
        T_ref_uint8.copy_(T_ref)
        torch.testing.assert_close(T_uint8, T_ref_uint8, rtol=0.0, atol=0.0)
        print("Checked FP8 tensors equal to each other")
        return

    assert not torch.isnan(T).any().item()
    assert not torch.isnan(T_ref).any().item()
    assert not torch.isinf(T).any().item()
    assert not torch.isinf(T_ref).any().item()
    sum_val = torch.sum(T).item()
    sum_val_ref = torch.sum(T_ref).item()
    max_abs_diff = torch.max(torch.abs(T - T_ref)).item()
    print(
        f"Sum val: {sum_val} vs. sum val ref: {sum_val_ref}, max diff: "
        f"{max_abs_diff}"
    )
    torch.testing.assert_close(T, T_ref, rtol=rtol, atol=atol)


@unittest.skipIf(not torch.cuda.is_available(), "Needs to run on a GPU")
class CustomOpTest(unittest.TestCase):
    @settings(verbosity=VERBOSITY, max_examples=1, deadline=None)
    @given(
        num_rows=st.sampled_from([8192]),
        num_cols=st.sampled_from([14336]),
    )
    def test_rescale(self, num_rows: int, num_cols: int) -> None:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.set_default_dtype(torch.bfloat16)

        # Create random tensors
        inputs = _get_tensor([num_rows, num_cols], torch.bfloat16, device)
        row_scale = _get_tensor([num_rows], torch.float32, device) + 1.0
        col_scale = _get_tensor([num_cols], torch.float32, device) + 1.0

        outputs_ref = torch.div(inputs, torch.outer(row_scale, col_scale)).to(
            torch.bfloat16
        )
        outputs = torch.empty_like(inputs)
        outputs = torch.ops.fbgemm.row_col_rescale(
            inputs, row_scale, col_scale, output=outputs
        )
        _check_tensor_all_close(outputs_ref, outputs)
