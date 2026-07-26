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
from hypothesis import given, settings, Verbosity

from .common import additional_decorators, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import optests, use_cpu_strategy
else:
    from fbgemm_gpu.test.test_utils import optests, use_cpu_strategy


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class JaggedIndexSelect2DTest(unittest.TestCase):
    @staticmethod
    def jagged_index_select_2d_ref(
        values: torch.Tensor,
        lengths: torch.Tensor,
        inverse_lookup: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        offsets = torch.ops.fbgemm.asynchronous_exclusive_cumsum(lengths)
        end_offsets = offsets + lengths
        full_start_offset = torch.index_select(offsets, 0, inverse_lookup)
        full_end_offset = torch.index_select(end_offsets, 0, inverse_lookup)
        index_ranges = torch.stack(
            (full_start_offset, full_end_offset), dim=0
        ).transpose(0, 1)

        to_be_merged_tensors = []
        for row in index_ranges:
            to_be_merged_tensors.append(torch.arange(row[0], row[1], device=device))
        all_indices = torch.cat(to_be_merged_tensors, dim=0)
        new_embeddings = torch.index_select(values, 0, all_indices)
        return new_embeddings

    @given(
        max_seq_length=st.integers(5, 10),
        batch_size=st.integers(1, 128),
        num_cols=st.integers(1, 128),
        num_jagged_tensor_rows=st.integers(1, 128),
        index_dtype=st.sampled_from([torch.int, torch.long]),
        jagged_tensor_dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.int,
                torch.long,
            ]  # Disable torch.bfloat16 due to large error bound
        ),
        use_cpu=use_cpu_strategy(),
        check_non_contiguous=st.booleans(),
        known_shape=st.booleans(),
    )
    @settings(max_examples=20, deadline=None, verbosity=Verbosity.normal)
    def test_jagged_index_select_2d(
        self,
        max_seq_length: int,
        batch_size: int,
        num_cols: int,
        num_jagged_tensor_rows: int,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        use_cpu: bool,
        check_non_contiguous: bool,
        known_shape: bool,
    ) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        is_float = jagged_tensor_dtype in [torch.float, torch.half, torch.bfloat16]
        lengths = torch.randint(
            low=0,
            high=max_seq_length,
            size=(num_jagged_tensor_rows,),
            dtype=index_dtype,
            device=device,
        )
        indices, _ = torch.sort(
            torch.randint(
                low=0,
                high=num_jagged_tensor_rows,
                size=(batch_size,),
                dtype=index_dtype,
                device=device,
            )
        )
        if is_float:
            values = torch.rand(
                int(lengths.sum().item()),
                num_cols,
                dtype=jagged_tensor_dtype,
                device=device,
            )
        else:
            values = torch.randint(
                2**16,
                (int(lengths.sum().item()), num_cols),
                dtype=jagged_tensor_dtype,
                device=device,
            )
        values_ref = values.detach().clone()

        if check_non_contiguous:
            values = values.as_strided(values.shape, (1, values.shape[0]))
            values_ref = values_ref.as_strided(values.shape, (1, values.shape[0]))

        # Only float tensors can require grad
        if is_float:
            values.requires_grad = True
            values_ref.requires_grad = True

        if known_shape:
            with torch.no_grad():
                tmp_output, _ = torch.ops.fbgemm.jagged_index_select(
                    values, lengths, indices
                )
            num_dense_output_rows = tmp_output.shape[0]
            output, _ = torch.ops.fbgemm.jagged_index_select(
                values, lengths, indices, num_dense_output_rows
            )
        else:
            output, _ = torch.ops.fbgemm.jagged_index_select(values, lengths, indices)
        output_ref = self.jagged_index_select_2d_ref(
            values_ref, lengths, indices, device
        )

        assert torch.equal(output, output_ref)

        if not is_float:
            return

        grad = torch.rand_like(output)
        grad_ref = grad.detach().clone()

        if check_non_contiguous:
            grad = grad.as_strided(grad.shape, (1, grad.shape[0]))
            grad_ref = grad_ref.as_strided(grad.shape, (1, grad.shape[0]))

        output.backward(grad)
        output_ref.backward(grad_ref)

        torch.testing.assert_close(
            values.grad,
            values_ref.grad,
            rtol=1e-2 if jagged_tensor_dtype in [torch.half, torch.bfloat16] else None,
            atol=1e-2 if jagged_tensor_dtype in [torch.half, torch.bfloat16] else None,
        )
        if known_shape:
            with torch.no_grad():
                tmp_output, _ = torch.ops.fbgemm.jagged_index_select(
                    values, lengths, indices
                )
            num_dense_output_rows = tmp_output.shape[0]
            torch.library.opcheck(
                torch.ops.fbgemm.jagged_index_select.default,
                (
                    values.detach().requires_grad_(),
                    lengths,
                    indices,
                    num_dense_output_rows,
                ),
            )
        else:
            torch.library.opcheck(
                torch.ops.fbgemm.jagged_index_select.default,
                (values.detach().requires_grad_(), lengths, indices),
            )

    @given(
        max_seq_length=st.integers(5, 10),
        batch_size=st.integers(1, 128),
        num_cols=st.integers(1, 128),
        num_jagged_tensor_rows=st.integers(1, 128),
        index_dtype=st.sampled_from([torch.int, torch.long]),
        jagged_tensor_dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.int,
                torch.long,
            ]  # Disable torch.bfloat16 due to large error bound
        ),
        use_cpu=use_cpu_strategy(),
    )
    @settings(max_examples=20, deadline=None)
    def test_jagged_index_select_2d_in_inference(
        self,
        max_seq_length: int,
        batch_size: int,
        num_cols: int,
        num_jagged_tensor_rows: int,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        use_cpu: bool,
    ) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        is_float = jagged_tensor_dtype in [torch.float, torch.half, torch.bfloat16]
        lengths = torch.randint(
            low=0,
            high=max_seq_length,
            size=(num_jagged_tensor_rows,),
            dtype=index_dtype,
            device=device,
        )
        indices, _ = torch.sort(
            torch.randint(
                low=0,
                high=num_jagged_tensor_rows,
                size=(batch_size,),
                dtype=index_dtype,
                device=device,
            )
        )
        if is_float:
            values = torch.rand(
                int(lengths.sum().item()),
                num_cols,
                dtype=jagged_tensor_dtype,
                device=device,
            )
        else:
            values = torch.randint(
                2**16,
                (int(lengths.sum().item()), num_cols),
                dtype=jagged_tensor_dtype,
                device=device,
            )
        values_ref = values.detach().clone()

        with torch.inference_mode():
            output, _ = torch.ops.fbgemm.jagged_index_select(values, lengths, indices)
            output_ref = self.jagged_index_select_2d_ref(
                values_ref, lengths, indices, device
            )
            assert torch.equal(output, output_ref)

    @optests.dontGenerateOpCheckTests("regression test for negative-size guard")
    def test_jagged_index_add_2d_forward_negative_rows_errors(self) -> None:
        """Regression: a negative num_output_rows must fail fast instead of
        allocating a tensor with a negative dimension."""
        device = torch.device("cpu")
        values = torch.zeros((2, 3), dtype=torch.float, device=device)
        indices = torch.tensor([0], dtype=torch.long, device=device)
        input_offsets = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
        output_offsets = torch.tensor([0, 1], dtype=torch.long, device=device)
        with self.assertRaisesRegex(ValueError, "num_output_rows must be non-negative"):
            torch.ops.fbgemm.jagged_index_add_2d_forward(
                values, indices, input_offsets, output_offsets, 1, -1
            )


class JaggedIndexSelect2DLargeGridTest(unittest.TestCase):
    """
    Retro: regression tests for the HIP grid-overflow bug in
    ``index_add_2d_with_unique_indices_kernel`` (D105029511 /
    Subplan B Diff #10), which lacked its own test method when
    landed.

    Block: dim3(stride_D / UNROLL_FACTOR, num_y_blocks).
    Grid: dim3(num_unique_indices, ceil(D / stride_D), 1).
    The production cap is `blocks_x = min(num_unique_indices,
    get_max_thread_blocks(stream))` (~16384 on MI300/MI350); the
    kernel grid-strides over the unique-index axis post-fix.
    """

    @classmethod
    def _has_gpu(cls) -> bool:
        return torch.cuda.is_available()

    @classmethod
    def _gpu_memory_lt(cls, gb: int) -> bool:
        if not cls._has_gpu():
            return True
        return torch.cuda.get_device_properties(0).total_memory < gb * (1 << 30)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
    def test_index_add_2d_with_unique_indices_correctness(self) -> None:
        """
        Multi-block correctness check at small scale via the autograd
        backward of ``jagged_index_select`` (which dispatches to
        ``index_add_2d_with_unique_indices_kernel``). Sentinel non-zero
        values at start / middle / end of the unique-index axis force
        the grid-stride outer loop to iterate.
        """
        if self._gpu_memory_lt(4):
            self.skipTest("Requires >= 4 GiB GPU memory")
        device = torch.accelerator.current_accelerator()
        # num_unique_indices > 2 * 1024 so the grid-stride loop iterates.
        N = 2 * 1024 + 3
        D = 16
        # Sparse lengths: most entries 0, sentinel non-zero at start /
        # middle / end so the kernel produces non-trivial backward grad.
        lengths_cpu = torch.zeros(N, dtype=torch.int64)
        lengths_cpu[0] = 1
        lengths_cpu[N // 2] = 2
        lengths_cpu[N - 1] = 3
        total = int(lengths_cpu.sum().item())
        # All unique inverse_lookup values so dedup keeps every batch.
        inverse_lookup_cpu = torch.arange(N, dtype=torch.int64)

        values_init = torch.arange(total * D, dtype=torch.float32).reshape(total, D)

        # GPU forward + backward.
        values_gpu = values_init.detach().clone().to(device).requires_grad_(True)
        output_gpu, _ = torch.ops.fbgemm.jagged_index_select(
            values_gpu, lengths_cpu.to(device), inverse_lookup_cpu.to(device)
        )
        output_gpu.sum().backward()

        # CPU reference: backward of jagged_index_select with a permutation
        # `inverse_lookup` is a scatter_add of grad over unique indices.
        # With identity inverse_lookup and grad = ones, the expected
        # gradient is `ones` for every selected row.
        # pyre-ignore[16]
        self.assertEqual(values_gpu.grad.shape, values_init.shape)
        torch.testing.assert_close(
            values_gpu.grad.cpu(),
            torch.ones_like(values_init),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
    def test_index_add_2d_with_unique_indices_large_grid(self) -> None:
        """
        Launch-survival regression test at the cap-trip scale.

        Pre-fix, ``index_add_2d_with_unique_indices_kernel`` launches
        with grid_x = num_unique_indices and per-block thread count
        determined by stride_D / UNROLL_FACTOR. With D = 8 and
        num_unique_indices = (1 << 22) + 1 the cap-trip path on ROCm
        would TORCH_CHECK-fail; post-fix the host caps grid_x to
        ``get_max_thread_blocks(stream)`` and the kernel grid-strides.

        Memory budget: values ~ N * D * 4B = 128 MiB per copy;
        the fwd output is the same size. Skip if HBM < 4 GiB.
        """
        if self._gpu_memory_lt(4):
            self.skipTest("Requires >= 4 GiB GPU memory")
        device = torch.accelerator.current_accelerator()
        N = (1 << 22) + 1
        D = 8
        # All-zero lengths with one non-zero entry so the backward
        # kernel still launches over all unique indices.
        lengths_cpu = torch.zeros(N, dtype=torch.int64)
        lengths_cpu[0] = 1
        total = int(lengths_cpu.sum().item())
        inverse_lookup_cpu = torch.arange(N, dtype=torch.int64)

        values = torch.zeros(
            (total, D), dtype=torch.float32, device=device, requires_grad=True
        )
        # Pre-fix this trips KernelLauncher::checkThreadCountNotExceeded
        # on ROCm at the index_add_2d_with_unique_indices launch.
        output, _ = torch.ops.fbgemm.jagged_index_select(
            values, lengths_cpu.to(device), inverse_lookup_cpu.to(device)
        )
        output.sum().backward()
        # pyre-ignore[16]
        self.assertEqual(values.grad.shape, values.shape)


if __name__ == "__main__":
    unittest.main()
