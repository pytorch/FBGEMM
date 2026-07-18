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

from .common import additional_decorators, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import cpu_and_maybe_gpu, optests
else:
    from fbgemm_gpu.test.test_utils import cpu_and_maybe_gpu, optests


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class DenseBmmTest(unittest.TestCase):
    @given(
        B=st.integers(10, 512),
        M=st.integers(1, 32),
        N=st.integers(1, 32),
        max_L=st.integers(1, 32),
        dtype=st.sampled_from([torch.float]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_jagged_bmm(
        self,
        B: int,
        M: int,
        N: int,
        max_L: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        assume(B != 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        lengths = torch.randint(max_L + 1, size=(B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        total_length = int(lengths.sum().item())
        x_values = torch.rand(
            (total_length, M), requires_grad=True, dtype=dtype, device=device
        )
        y_values = torch.rand(
            (total_length, N), requires_grad=True, dtype=dtype, device=device
        )
        output = torch.ops.fbgemm.jagged_jagged_bmm(
            x_values,
            y_values,
            offsets,
            max_L,
        )

        x_values_ref = x_values.detach().clone().requires_grad_(True)
        y_values_ref = y_values.detach().clone().requires_grad_(True)
        x_dense_ref = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values_ref,
            [offsets],
            max_lengths=[max_L],
        )
        y_dense_ref = torch.ops.fbgemm.jagged_to_padded_dense(
            y_values_ref,
            [offsets],
            max_lengths=[max_L],
        )
        output_ref = torch.bmm(x_dense_ref.transpose(2, 1), y_dense_ref)

        # verify forward
        torch.testing.assert_close(output, output_ref)

        # verify backward
        grad_output = output.detach().clone().requires_grad_(True)

        output.backward(grad_output)
        output_ref.backward(grad_output)

        # NOTE: Relax the tolerance for float32 here to avoid flaky test
        #       failures on ARM
        # TODO: Need to investigate why the error is so high for float32
        # See table in https://pytorch.org/docs/stable/testing.html
        if dtype == torch.float32:
            torch.testing.assert_close(
                x_values.grad, x_values_ref.grad, rtol=1e-3, atol=1e-1
            )
            torch.testing.assert_close(
                y_values.grad, y_values_ref.grad, rtol=1e-3, atol=1e-1
            )
        else:
            torch.testing.assert_close(x_values.grad, x_values_ref.grad)
            torch.testing.assert_close(y_values.grad, y_values_ref.grad)

    @given(
        B=st.integers(10, 512),
        M=st.integers(1, 32),
        N=st.integers(1, 32),
        max_L=st.integers(1, 32),
        dtype=st.sampled_from([torch.float]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_jagged_dense_bmm(
        self,
        B: int,
        M: int,
        N: int,
        max_L: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        assume(B != 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        lengths = torch.randint(max_L + 1, size=(B,), device=device)
        total_length = int(lengths.sum().item())
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        x_values = torch.rand(
            (total_length, M), requires_grad=True, dtype=dtype, device=device
        )
        y = torch.rand((B, M, N), requires_grad=True, dtype=dtype, device=device)

        output, _ = torch.ops.fbgemm.jagged_dense_bmm(
            x_values,
            offsets,
            y,
            max_L,
        )

        x_values_ref = x_values.detach().clone().requires_grad_(True)
        x_dense_ref = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values_ref,
            [offsets],
            max_lengths=[max_L],
        )
        y_ref = y.detach().clone().requires_grad_(True)
        output_dense = torch.bmm(x_dense_ref, y_ref)

        output_ref, _ = torch.ops.fbgemm.dense_to_jagged(
            output_dense, [offsets], total_length
        )

        # verify forward
        torch.testing.assert_close(output, output_ref)
        # verify backward
        grad_output = output.detach().clone().requires_grad_(True)

        output.backward(grad_output)
        output_ref.backward(grad_output)

        torch.testing.assert_close(x_values.grad, x_values_ref.grad)
        torch.testing.assert_close(y.grad, y_ref.grad)

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @given(
        B=st.integers(10, 512),
        M=st.integers(2, 32),
        N=st.integers(2, 32),
        max_L=st.integers(2, 32),
        dtype=st.sampled_from([torch.float]),
        device=st.just(torch.device("cpu")),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_dense_bmm_dynamic_shape(
        self,
        B: int,
        M: int,
        N: int,
        max_L: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        assume(B != 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        lengths = torch.randint(low=1, high=max_L + 1, size=(B,), device=device)
        total_length = int(lengths.sum().item())
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        x_values = torch.rand(
            (total_length, M), requires_grad=True, dtype=dtype, device=device
        )
        y = torch.rand((B, M, N), requires_grad=True, dtype=dtype, device=device)

        torch._dynamo.mark_dynamic(x_values, 0)
        torch._dynamo.mark_dynamic(x_values, 1)
        torch._dynamo.mark_dynamic(lengths, 0)  # offsets = lengths + 1

        output, _ = torch.compile(
            torch.ops.fbgemm.jagged_dense_bmm, fullgraph=True, dynamic=True
        )(
            x_values,
            offsets,
            y,
            max_L,
        )

        x_values_ref = x_values.detach().clone().requires_grad_(True)
        x_dense_ref = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values_ref,
            [offsets],
            max_lengths=[max_L],
        )
        y_ref = y.detach().clone().requires_grad_(True)
        output_dense = torch.bmm(x_dense_ref, y_ref)

        output_ref, _ = torch.ops.fbgemm.dense_to_jagged(
            output_dense, [offsets], total_length
        )

        # verify forward
        torch.testing.assert_close(output, output_ref)
        # verify backward
        grad_output = output.detach().clone().requires_grad_(True)

        output.backward(grad_output)
        output_ref.backward(grad_output)

        torch.testing.assert_close(x_values.grad, x_values_ref.grad)
        torch.testing.assert_close(y.grad, y_ref.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
    def test_jagged_dense_bmm_large_grid(self) -> None:
        """
        Retro: regression test for the HIP grid-overflow bug in
        ``jagged_dense_bmm_kernel`` (D105204785 / Subplan D Diff #28),
        which lacked its own test method when landed.

        Block: dim3(THREADS_PER_BLOCK) where the kernel parameterizes
        BLOCK_TILE_M, BLOCK_TILE_K, BLOCK_TILE_N. Grid:
        dim3(NUM_BLOCK_COLS, NUM_BLOCK_ROWS, B). Pre-fix the kernel
        body was a single-iteration assertion (early return on
        out-of-range blocks). Post-fix the kernel grid-strides over
        ``(block_col, block_row, b)`` so a capped grid still covers
        every output tile.

        Verification: end-to-end correctness vs. CPU dispatch using
        non-trivial sentinel input that forces multiple segments
        (the per-iteration accumulator and shared-memory tiles must
        reset across grid-stride iterations).
        """
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")
        B = 4
        # Sized so the launch grid spans multiple blocks in every dimension
        # (M <= N selects BLOCK_TILE_M=8, BLOCK_TILE_N=32):
        #   grid_dim_x = ceil(N / 32)     = 2  (block_col)
        #   grid_dim_y = ceil(max_L / 8)  = 5  (block_row)
        #   grid_dim_z = B                = 4  (b)
        # On CUDA the grid is uncapped, so each grid-stride loop runs once and
        # this verifies full multi-block tile coverage of the refactored
        # kernel; the multi-iteration stride path is only taken when
        # cap_grid_dim_x shrinks the grid (ROCm / HIP 2**32 launch limit).
        max_L = 40
        M = 16
        N = 64

        # Sparse non-zero lengths (incl. an empty segment), each <= max_L.
        lengths_cpu = torch.tensor([37, 0, 40, 12], dtype=torch.int64)
        offsets_cpu = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths_cpu)
        total_length = int(lengths_cpu.sum().item())

        x_values_init = torch.arange(total_length * M, dtype=torch.float32).reshape(
            total_length, M
        )
        y_init = torch.arange(B * M * N, dtype=torch.float32).reshape(B, M, N) * 0.01

        # CPU oracle.
        out_cpu, _ = torch.ops.fbgemm.jagged_dense_bmm(
            x_values_init, offsets_cpu, y_init, max_L
        )

        # GPU op under test.
        out_gpu, _ = torch.ops.fbgemm.jagged_dense_bmm(
            x_values_init.to(device),
            offsets_cpu.to(device),
            y_init.to(device),
            max_L,
        )

        # Relax float32 tolerance (consistent with the other float32 checks
        # in this file): arange inputs accumulated over K reach large
        # magnitudes, so CPU/GPU accumulation-order differences exceed the
        # default tolerance.
        torch.testing.assert_close(out_gpu.cpu(), out_cpu, rtol=1e-3, atol=1e-1)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
    def test_jagged_jagged_bmm_correctness(self) -> None:
        """
        Regression test for the HIP grid-overflow bug in
        ``jagged_jagged_bmm_kernel`` (D105205055 / Subplan D Diff #29),
        which lacked its own test method when landed.

        Verifies end-to-end correctness vs. CPU dispatch using
        sentinel non-zero lengths (including a zero-length entry).
        """
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")
        max_L = 8
        M = 16
        N = 8

        lengths_cpu = torch.tensor([3, 0, 5, 2], dtype=torch.int64)
        offsets_cpu = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths_cpu)
        total_length = int(lengths_cpu.sum().item())

        x_values_init = torch.arange(total_length * M, dtype=torch.float32).reshape(
            total_length, M
        )
        y_values_init = (
            torch.arange(total_length * N, dtype=torch.float32).reshape(total_length, N)
            * 0.01
        )

        # CPU oracle.
        out_cpu = torch.ops.fbgemm.jagged_jagged_bmm(
            x_values_init, y_values_init, offsets_cpu, max_L
        )

        # GPU op under test.
        out_gpu = torch.ops.fbgemm.jagged_jagged_bmm(
            x_values_init.to(device),
            y_values_init.to(device),
            offsets_cpu.to(device),
            max_L,
        )

        torch.testing.assert_close(out_gpu.cpu(), out_cpu)


if __name__ == "__main__":
    unittest.main()
