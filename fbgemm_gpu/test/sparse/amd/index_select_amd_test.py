#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest

import fbgemm_gpu.sparse_ops  # noqa: F401
import torch

# Smallest embedding-row count the filler indices below are valid for.
MIN_EMBEDDING_ROWS = 6


def _make_unsorted_indices(
    duplicate_run_length: int,
    group_id: int,
    index_dtype: torch.dtype,
) -> torch.Tensor:
    """Build unsorted indices with a known duplicate run after sorting.

    The filler values deliberately skip rows so that, once sorted, at least one
    zero-length row sits BETWEEN two populated rows. That is the interior case
    for find_segment(): it returns first - 1 from an upper-bound scan, so a
    chunk boundary landing on a duplicate offset (chunk_offsets[r] ==
    chunk_offsets[r + 1]) must still resolve to the right row. Skipped rows only
    at the tail never own a chunk and would leave that branch uncovered.

    Requires num_embedding_rows >= MIN_EMBEDDING_ROWS.
    """
    num_fillers = 4
    indices = torch.zeros(duplicate_run_length + num_fillers, dtype=index_dtype)
    filler_positions = [
        0,
        indices.numel() // 3,
        indices.numel() // 2,
        indices.numel() - 1,
    ]
    filler_values = [
        2 + group_id % 2,
        1,
        5,
        1,
    ]
    indices[filler_positions] = torch.tensor(filler_values, dtype=index_dtype)
    return indices


def _make_grad_output(
    num_indices: int,
    num_cols: int,
    dtype: torch.dtype,
    group_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target-dtype gradients and their exact FP64 values."""
    grad = torch.full((num_indices, num_cols), 0.125, dtype=dtype)
    filler_positions = [0, num_indices // 3, num_indices // 2, num_indices - 1]
    cols = torch.arange(num_cols, dtype=dtype)
    for filler_id, position in enumerate(filler_positions):
        grad[position] = (
            0.25 + (group_id * 0.125) + (filler_id * 0.125) + cols.remainder(4)
        )
    return grad, grad.to(torch.float64)


def _grad_tolerance(dtype: torch.dtype, num_indices: int) -> float:
    """Relative tolerance derived from `dtype`, not from a flat constant.

    The kernel accumulates in FP32 (at::acc_type) and rounds once into `dtype`,
    so a correct result lands within a ULP of an FP64 reference plus whatever the
    FP32 accumulation order costs over the duplicate run. Deriving the bound from
    the format matters most for BF16, whose own resolution is ~0.4%: a flat 1e-2
    is over two ULPs and can absorb several dropped contributions.
    """
    return (
        torch.finfo(dtype).eps + math.sqrt(num_indices) * torch.finfo(torch.float32).eps
    )


def _unselected_rows_mask(indices: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Rows that no index selects, derived from the indices themselves."""
    selected = torch.zeros(num_rows, dtype=torch.bool)
    selected[indices.long()] = True
    return ~selected


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None,
    "ROCm GPU required",
)
class IndexSelectAmdTest(unittest.TestCase):
    def test_group_index_select_backward_col_tile_boundary(self) -> None:
        """Exercise the ROCm contiguous-warp cache at a column-tile boundary."""
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")
        dtype = torch.float
        num_cols = 64
        num_embedding_rows = 10
        num_indices = 100003

        # Citrine C3: create the regression inputs directly on the AMD device.
        indices = torch.zeros(num_indices, device=device, dtype=torch.long)
        input_tensor = torch.randn(
            num_embedding_rows, num_cols, device=device, dtype=dtype
        ).requires_grad_(True)
        input_ref = input_tensor.detach().clone().requires_grad_(True)

        output = torch.ops.fbgemm.group_index_select_dim0([input_tensor], [indices])
        output_ref = [torch.index_select(input_ref, 0, indices)]
        grad = torch.ones(num_indices, num_cols, device=device, dtype=dtype)
        output_ref[0].backward(grad)
        output[0].backward(grad)

        torch.testing.assert_close(
            input_tensor.grad,
            input_ref.grad,
            atol=0.5,
            rtol=0,
            msg="grad_input mismatch at the ROCm column-tile cache boundary",
        )

    def _assert_grad_matches_reference(
        self,
        grad: torch.Tensor,
        expected_fp64: torch.Tensor,
        indices_cpu: torch.Tensor,
        dtype: torch.dtype,
    ) -> None:
        """Compare a flattened grad against an FP64 reference, then check zeros.

        expected_fp64 is NOT downcast to `dtype`: rounding the reference first
        throws away the precision the comparison is supposed to be measuring.
        """
        num_rows, _ = expected_fp64.shape
        tolerance = _grad_tolerance(dtype, indices_cpu.numel())
        torch.testing.assert_close(
            grad.double(),
            expected_fp64,
            atol=tolerance * float(expected_fp64.abs().max()),
            rtol=tolerance,
        )
        unselected = _unselected_rows_mask(indices_cpu, num_rows)
        if bool(unselected.any()):
            torch.testing.assert_close(
                grad[unselected].double(),
                torch.zeros_like(expected_fp64[unselected]),
                atol=0,
                rtol=0,
                msg="rows selected by no index must have exactly zero gradient",
            )

    def _assert_group_index_select_backward(
        self,
        duplicate_run_length: int,
        widths: list[int],
        dtype: torch.dtype,
        num_embedding_rows_group: list[int] | None = None,
        index_dtype: torch.dtype = torch.long,
        trailing_shapes: list[tuple[int, ...]] | None = None,
    ) -> None:
        """Run one grouped backward and check every group's gradient.

        `widths` are FLATTENED column counts. `trailing_shapes` optionally gives
        the N-D trailing dims behind each width, exercising the kernel's
        reliance on input.reshape({input.size(0), -1}) to flatten them.
        """
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")
        if num_embedding_rows_group is None:
            num_embedding_rows_group = [8] * len(widths)
        self.assertEqual(len(widths), len(num_embedding_rows_group))
        if trailing_shapes is None:
            trailing_shapes = [(width,) for width in widths]
        self.assertEqual(len(widths), len(trailing_shapes))
        input_group: list[torch.Tensor] = []
        indices_group: list[torch.Tensor] = []
        grad_group: list[torch.Tensor] = []
        expected_group: list[torch.Tensor] = []
        indices_cpu_group: list[torch.Tensor] = []

        for group_id, (num_cols, num_embedding_rows, trailing) in enumerate(
            zip(widths, num_embedding_rows_group, trailing_shapes)
        ):
            self.assertGreaterEqual(num_embedding_rows, MIN_EMBEDDING_ROWS)
            self.assertEqual(math.prod(trailing), num_cols)
            indices_cpu = _make_unsorted_indices(
                duplicate_run_length,
                group_id,
                index_dtype,
            )
            num_indices = indices_cpu.numel()
            grad_cpu, grad_fp64 = _make_grad_output(
                num_indices, num_cols, dtype, group_id
            )
            expected = torch.zeros(
                num_embedding_rows, num_cols, dtype=torch.float64
            ).index_add_(0, indices_cpu.long(), grad_fp64)

            input_group.append(
                torch.randn(
                    num_embedding_rows,
                    *trailing,
                    device=device,
                    dtype=dtype,
                ).requires_grad_(True)
            )
            indices_group.append(indices_cpu.to(device))
            grad_group.append(grad_cpu.reshape(num_indices, *trailing).to(device))
            expected_group.append(expected)
            indices_cpu_group.append(indices_cpu)

        output_group = torch.ops.fbgemm.group_index_select_dim0(
            input_group, indices_group
        )
        torch.autograd.backward(output_group, grad_group)

        for group_id, (input_tensor, expected, indices_cpu) in enumerate(
            zip(input_group, expected_group, indices_cpu_group)
        ):
            with self.subTest(group_id=group_id, num_cols=widths[group_id]):
                self.assertIsNotNone(input_tensor.grad)
                assert input_tensor.grad is not None
                self._assert_grad_matches_reference(
                    input_tensor.grad.cpu().reshape(expected.shape),
                    expected,
                    indices_cpu,
                    dtype,
                )

    def test_group_index_select_backward_segment_chunk_boundaries(self) -> None:
        """Sweep ONLY the duplicate run length across the 512-wide chunk bound.

        Width and dtype are held fixed so a failure here localises to the chunk
        arithmetic; dtype and width get their own sweeps below.
        """
        for duplicate_run_length in [511, 512, 513, 1025]:
            with self.subTest(duplicate_run_length=duplicate_run_length):
                self._assert_group_index_select_backward(
                    duplicate_run_length, [64], torch.float
                )

    def test_group_index_select_backward_dtypes(self) -> None:
        """Sweep ONLY dtype, at a fixed multi-chunk run length and width."""
        for dtype in [torch.float, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                self._assert_group_index_select_backward(513, [64], dtype)

    def test_group_index_select_backward_widths(self) -> None:
        """Sweep ONLY the column width, including non-powers of two."""
        for num_cols in [32, 64, 65, 127, 128]:
            with self.subTest(num_cols=num_cols):
                self._assert_group_index_select_backward(513, [num_cols], torch.float)

    def test_group_index_select_backward_segment_branch_boundary(self) -> None:
        """kMinSegmentReduceDupFactor = 32: 31 falls back, 32 is eligible."""
        num_embedding_rows = 8
        num_fillers = 4
        for duplicate_factor in [31, 32]:
            with self.subTest(duplicate_factor=duplicate_factor):
                self._assert_group_index_select_backward(
                    duplicate_run_length=(duplicate_factor * num_embedding_rows)
                    - num_fillers,
                    widths=[64],
                    dtype=torch.float,
                    num_embedding_rows_group=[num_embedding_rows],
                )

    def test_group_index_select_backward_segment_cols_boundary(self) -> None:
        """kMinSegmentReduceCols = 32: 31 falls back, 32 is eligible.

        Mirrors the duplication-factor boundary above for the other gate term.
        The duplication factor is held well above its own threshold so that the
        column width is the only thing deciding which path runs.
        """
        for num_cols in [31, 32]:
            with self.subTest(num_cols=num_cols):
                self._assert_group_index_select_backward(513, [num_cols], torch.float)

    def test_group_index_select_backward_nd_input(self) -> None:
        """N-D inputs: trailing dims are flattened into num_cols_group.

        The kernel reads grad_output[source_row * num_cols + col] with num_cols
        from input.reshape({input.size(0), -1}).size(1), so an (8, 8, 16) input
        must behave exactly like a flattened 128-wide one.
        """
        self._assert_group_index_select_backward(
            duplicate_run_length=513,
            widths=[128],
            dtype=torch.float,
            trailing_shapes=[(8, 16)],
        )

    def test_group_index_select_backward_multiple_groups(self) -> None:
        self._assert_group_index_select_backward(
            duplicate_run_length=513,
            widths=[32, 65, 128],
            dtype=torch.bfloat16,
            num_embedding_rows_group=[8, 9, 10],
            index_dtype=torch.int32,
        )

    def test_group_index_select_backward_non_contiguous_inputs(self) -> None:
        """Non-contiguous grad and indices must still produce exact gradients.

        The ROCm segment-reduce gate deliberately does not test the caller's
        layout: expect_contiguous() runs unconditionally, and the kernel consumes
        grad_output_contigs and the sort outputs, never the tensors passed in
        here. The grad shape below is the one that motivated dropping those
        guards -- the backward of a torch.cat(..., dim=1) hands each input a
        column slice whose row stride is the full concatenated width.

        Sized to land in the segment path: 1029 indices over 8 rows is a
        duplication factor of 128 (gate needs >= 32), 64 cols (needs >= 32), and
        1029 spans three 512-wide chunks so the multi-chunk atomic branch runs.
        """
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")
        dtype = torch.float
        num_cols = 64
        num_embedding_rows = 8
        indices_cpu = _make_unsorted_indices(1025, 0, torch.long)
        num_indices = indices_cpu.numel()
        grad_cpu, grad_fp64 = _make_grad_output(num_indices, num_cols, dtype, 0)
        expected = torch.zeros(
            num_embedding_rows, num_cols, dtype=torch.float64
        ).index_add_(0, indices_cpu, grad_fp64)

        for name, noncontig_grad, noncontig_indices in [
            ("grad", True, False),
            ("indices", False, True),
            ("both", True, True),
        ]:
            with self.subTest(name=name):
                if noncontig_grad:
                    # Middle column slice of a 3x-wide parent: strides
                    # (3 * num_cols, 1), so neither side of the slice is empty.
                    grad_parent = torch.zeros(
                        num_indices, num_cols * 3, device=device, dtype=dtype
                    )
                    grad = grad_parent[:, num_cols : num_cols * 2]
                    grad.copy_(grad_cpu.to(device))
                else:
                    grad = grad_cpu.to(device)

                if noncontig_indices:
                    # Every other element of a 2x-long parent: stride 2.
                    index_parent = torch.zeros(
                        num_indices * 2, device=device, dtype=torch.long
                    )
                    index_parent[::2] = indices_cpu.to(device)
                    indices = index_parent[::2]
                else:
                    indices = indices_cpu.to(device)

                # Guard against the case under test quietly disappearing.
                self.assertEqual(grad.is_contiguous(), not noncontig_grad)
                self.assertEqual(indices.is_contiguous(), not noncontig_indices)
                self.assertEqual(indices.numel(), grad.size(0))

                input_tensor = torch.randn(
                    num_embedding_rows, num_cols, device=device, dtype=dtype
                ).requires_grad_(True)
                output_group = torch.ops.fbgemm.group_index_select_dim0(
                    [input_tensor], [indices]
                )
                torch.autograd.backward(output_group, [grad])

                self.assertIsNotNone(input_tensor.grad)
                assert input_tensor.grad is not None
                self._assert_grad_matches_reference(
                    input_tensor.grad.cpu(), expected, indices_cpu, dtype
                )

    def test_group_index_select_backward_hot_row_accumulates_in_fp32(self) -> None:
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")
        num_indices = 4096
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                input_tensor = torch.zeros(
                    2, 64, device=device, dtype=dtype, requires_grad=True
                )
                indices = torch.zeros(num_indices, device=device, dtype=torch.long)
                contribution = torch.tensor(0.1, device=device, dtype=dtype)

                output = torch.ops.fbgemm.group_index_select_dim0(
                    [input_tensor], [indices]
                )[0]
                output.backward(contribution.expand_as(output).contiguous())

                expected = torch.zeros_like(input_tensor)
                expected[0] = contribution.float() * num_indices
                assert input_tensor.grad is not None
                torch.testing.assert_close(input_tensor.grad, expected, atol=0, rtol=0)
