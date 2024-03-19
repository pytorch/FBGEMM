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
    @settings(max_examples=20, deadline=None, verbosity=Verbosity.verbose)
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


if __name__ == "__main__":
    unittest.main()
