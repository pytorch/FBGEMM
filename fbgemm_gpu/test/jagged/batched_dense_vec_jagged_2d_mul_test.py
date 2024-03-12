#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest

import hypothesis.strategies as st
import torch
import torch._dynamo
from hypothesis import assume, given, settings, Verbosity

from .common import additional_decorators, open_source, torch_compiled

if open_source:
    # pyre-ignore[21]
    from test_utils import (
        cpu_and_maybe_gpu,
        gradcheck,
        optests,
        symint_vector_unsupported,
    )
else:
    from fbgemm_gpu.test.test_utils import (
        cpu_and_maybe_gpu,
        gradcheck,
        optests,
        symint_vector_unsupported,
    )


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class BatchedDenseVecJagged2DMulTest(unittest.TestCase):
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    @given(
        B=st.integers(0, 32),
        H=st.integers(1, 3),
        max_L=st.integers(1, 32),
        D=st.integers(0, 32),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=cpu_and_maybe_gpu(),
    )
    def test_batched_dense_vec_jagged_2d_mul(
        self,
        B: int,
        H: int,
        max_L: int,
        D: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        assume(H == 1 or B != 0)
        # CPU doesn't support bfloat16
        assume(device != torch.device("cpu") or dtype != torch.bfloat16)

        torch.backends.cuda.matmul.allow_tf32 = False

        # Sometimes length[i] exceed max_L meaning jagged->dense will be
        # truncation vs. padding
        lengths = torch.randint(max_L * 2, size=(B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        values = torch.rand((offsets[-1], H * D), dtype=dtype, device=device)
        dense = torch.rand((B * H, max_L), dtype=dtype, device=device)
        padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [offsets],
            [max_L],
        )  # [B, N, H * D]

        bmm_arg1 = dense.unsqueeze(1)
        bmm_arg2 = (
            padded_values.reshape(B, max_L, H, D)
            .transpose(1, 2)
            .reshape(B * H, max_L, D)
        )
        # torch.bmm not implemented for Half on CPU
        if dtype in [torch.half, torch.bfloat16] and device == torch.device("cpu"):
            bmm_arg1 = bmm_arg1.float()
            bmm_arg2 = bmm_arg2.float()
        output_ref = torch.bmm(bmm_arg1, bmm_arg2).squeeze(
            1
        )  # [B H, 1, N] x [B H, N, D] = [B H, 1, D]
        if dtype in [torch.half, torch.bfloat16] and device == torch.device("cpu"):
            output_ref = output_ref.to(dtype)
        output = torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul(
            dense, values, offsets
        )
        torch.testing.assert_close(
            output,
            output_ref,
            rtol=1e-2 if dtype in [torch.half, torch.bfloat16] else None,
            atol=1e-2 if dtype in [torch.half, torch.bfloat16] else None,
        )

        gradcheck(
            torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul,
            (
                dense.clone().detach().float().requires_grad_(True),
                values.clone().detach().float().requires_grad_(True),
                offsets,
            ),
            eps=1e-2,
            atol=1e-3,
            rtol=1e-3,
        )

    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    @given(
        B=st.integers(0, 32),
        H=st.integers(1, 3),
        max_L=st.integers(1, 32),
        D=st.integers(0, 32),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.sampled_from(["meta"]),
    )
    def test_batched_dense_vec_jagged_2d_mul_meta_backend(
        self,
        B: int,
        H: int,
        max_L: int,
        D: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        assume(H == 1 or B != 0)

        device = torch.device("cpu")
        torch.backends.cuda.matmul.allow_tf32 = False

        # Sometimes length[i] exceed max_L meaning jagged->dense will be
        # truncation vs. padding
        lengths = torch.randint(max_L * 2, size=(B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        values = torch.rand((offsets[-1], H * D), dtype=dtype, device=device)
        dense = torch.rand((B * H, max_L), dtype=dtype, device=device)
        padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [offsets],
            [max_L],
        )  # [B, N, H * D]

        bmm_arg1 = dense.unsqueeze(1)
        bmm_arg2 = (
            padded_values.reshape(B, max_L, H, D)
            .transpose(1, 2)
            .reshape(B * H, max_L, D)
        )
        # torch.bmm not implemented for Half on CPU
        if dtype in [torch.half, torch.bfloat16]:
            bmm_arg1 = bmm_arg1.float()
            bmm_arg2 = bmm_arg2.float()
        output_ref = torch.bmm(bmm_arg1, bmm_arg2).squeeze(
            1
        )  # [B H, 1, N] x [B H, N, D] = [B H, 1, D]
        dense.to(device_type)
        values.to(device_type)
        output = torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul(
            dense, values, offsets
        )
        assert output.size() == output_ref.size()

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @unittest.skipIf(*symint_vector_unsupported())
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    @given(
        B=st.integers(2, 32),
        H=st.integers(1, 3),
        max_L=st.integers(1, 32),
        D=st.integers(2, 32),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.just("cpu"),
    )
    def test_batched_dense_vec_jagged_2d_mul_dynamic_shape(
        self,
        B: int,
        H: int,
        max_L: int,
        D: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        assume(H == 1 or B != 0)

        device = torch.device(device_type)
        torch.backends.cuda.matmul.allow_tf32 = False

        # Sometimes length[i] exceed max_L meaning jagged->dense will be
        # truncation vs. padding
        lengths = torch.randint(low=1, high=max_L * 2, size=(B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        values = torch.rand((offsets[-1], H * D), dtype=dtype, device=device)
        dense = torch.rand((B * H, max_L), dtype=dtype, device=device)
        padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [offsets],
            [max_L],
        )  # [B, N, H * D]

        bmm_arg1 = dense.unsqueeze(1)
        bmm_arg2 = (
            padded_values.reshape(B, max_L, H, D)
            .transpose(1, 2)
            .reshape(B * H, max_L, D)
        )
        # torch.bmm not implemented for Half on CPU
        if dtype in [torch.half, torch.bfloat16]:
            bmm_arg1 = bmm_arg1.float()
            bmm_arg2 = bmm_arg2.float()
        output_ref = torch.bmm(bmm_arg1, bmm_arg2).squeeze(
            1
        )  # [B H, 1, N] x [B H, N, D] = [B H, 1, D]
        dense.to(device_type)
        values.to(device_type)

        torch._dynamo.mark_dynamic(dense, 0)
        torch._dynamo.mark_dynamic(values, 0)
        torch._dynamo.mark_dynamic(values, 1)
        torch._dynamo.mark_dynamic(offsets, 0)

        output = torch_compiled(
            torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul,
            fullgraph=True,
            dynamic=True,
        )(dense, values, offsets)
        assert output.size() == output_ref.size()


if __name__ == "__main__":
    unittest.main()
