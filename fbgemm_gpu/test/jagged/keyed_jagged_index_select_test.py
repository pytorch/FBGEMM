#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import random
import unittest

import hypothesis.strategies as st
import torch
import torch._dynamo
from hypothesis import given, settings

from .common import additional_decorators, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_memory_lt_gb, gpu_unavailable, optests, running_in_oss
else:
    from fbgemm_gpu.test.test_utils import (
        gpu_memory_lt_gb,
        gpu_unavailable,
        optests,
        running_in_oss,
    )


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class KeyedJaggedIndexSelectTest(unittest.TestCase):
    def _execute_keyed_jagged_index_select_dim1(
        self,
        max_seq_length: int,
        input_batch_size: int,
        output_batch_size: int,
        num_batches: int,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        has_weights: bool,
        check_non_contiguous: bool,
        use_selected_lengths_sum: bool,
    ) -> None:
        is_float = jagged_tensor_dtype in [torch.float, torch.half, torch.bfloat16]
        device = torch.accelerator.current_accelerator()
        lengths = torch.randint(
            low=0,
            high=max_seq_length,
            size=(input_batch_size * num_batches,),
            dtype=index_dtype,
            device=device,
        )
        offsets = torch.concat(
            [torch.zeros(1, dtype=torch.long, device=device), lengths.cumsum(0)]
        )
        indices = torch.randint(
            low=0,
            high=input_batch_size,
            size=(output_batch_size,),
            dtype=index_dtype,
            device=device,
        )

        # If check_non_contiguous=True, create a tensor that is twice as big
        # and then select only odd indices to make it non contiguous
        values_numel = int(offsets[-1].item())
        values_numel = values_numel * 2 if check_non_contiguous else values_numel

        if is_float:
            values = torch.rand(
                values_numel,
                dtype=jagged_tensor_dtype,
                device=device,
            )
        else:
            values = torch.randint(
                2**16,
                (values_numel,),
                dtype=jagged_tensor_dtype,
                device=device,
            )
        values_ref = values.detach().clone()

        if check_non_contiguous:
            values = values[1::2]
            values_ref = values_ref[1::2]

        if has_weights:
            weights = torch.rand(
                int(offsets[-1].item()),
                dtype=random.choice([torch.float, torch.half]),
                device=device,
            )
        else:
            weights = None

        if use_selected_lengths_sum:
            length_indices = torch.cat(
                [indices + i * input_batch_size for i in range(num_batches)]
            )
            selected_lengths_sum = (
                torch.index_select(lengths, 0, length_indices).sum().item()
            )
        else:
            selected_lengths_sum = None

        # Only float tensors can require grad
        if is_float:
            values.requires_grad = True
            values_ref.requires_grad = True

        index_select_output = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
            values,
            lengths,
            offsets,
            indices,
            input_batch_size,
            weights,
            selected_lengths_sum,
        )
        output = index_select_output[0]
        if has_weights:
            output_weights = index_select_output[2]

        output_ref = []
        output_weight_ref = []
        for k in range(num_batches):
            key_lengths = lengths[k * input_batch_size : (k + 1) * input_batch_size]
            start_offset = offsets[k * input_batch_size]
            end_offset = offsets[(k + 1) * input_batch_size]
            key_values = values_ref[start_offset:end_offset].view(-1, 1)
            output_ref.append(
                torch.ops.fbgemm.jagged_index_select(key_values, key_lengths, indices)[
                    0
                ].view(-1)
            )
            if has_weights:
                # pyre-ignore[16]
                key_weights = weights[start_offset:end_offset].view(-1, 1)
                output_weight_ref.append(
                    torch.ops.fbgemm.jagged_index_select(
                        key_weights, key_lengths, indices
                    )[0].view(-1)
                )

        output_ref = torch.concat(output_ref)
        assert torch.equal(output, output_ref)

        if has_weights:
            output_weight_ref = torch.concat(output_weight_ref)
            # pyre-ignore[61]
            assert torch.equal(output_weights, output_weight_ref)

        if not is_float:
            return

        # If check_non_contiguous=True, create a tensor that is twice as big
        # and then select only odd indices to make it non contiguous
        grad_numel = output.numel()
        grad_numel = grad_numel * 2 if check_non_contiguous else grad_numel

        grad = torch.rand(grad_numel, dtype=output.dtype, device=output.device)
        grad_ref = grad.detach().clone()

        if check_non_contiguous:
            grad = grad[1::2]
            grad_ref = grad_ref[1::2]

        output.backward(grad)
        output_ref.backward(grad_ref)

        torch.testing.assert_close(
            values.grad,
            values_ref.grad,
            rtol=1e-2 if jagged_tensor_dtype in [torch.half, torch.bfloat16] else None,
            atol=1e-2 if jagged_tensor_dtype in [torch.half, torch.bfloat16] else None,
        )

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_in_oss)
    @given(
        max_seq_length=st.integers(5, 10),
        input_batch_size=st.integers(1, 128),
        output_batch_size=st.integers(1, 128),
        num_batches=st.integers(1, 3),
        index_dtype=st.sampled_from([torch.int, torch.long]),
        jagged_tensor_dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.int,
                torch.long,
            ]  # Disable torch.bfloat16 due to large error bound
        ),
        has_weights=st.booleans(),
        check_non_contiguous=st.booleans(),
        use_selected_lengths_sum=st.booleans(),
    )
    @settings(max_examples=20, deadline=None)
    def test_keyed_jagged_index_select_dim1(
        self,
        max_seq_length: int,
        input_batch_size: int,
        output_batch_size: int,
        num_batches: int,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        has_weights: bool,
        check_non_contiguous: bool,
        use_selected_lengths_sum: bool,
    ) -> None:
        self._execute_keyed_jagged_index_select_dim1(
            max_seq_length,
            input_batch_size,
            output_batch_size,
            num_batches,
            index_dtype,
            jagged_tensor_dtype,
            has_weights,
            check_non_contiguous,
            use_selected_lengths_sum,
        )

    @given(
        index_dtype=st.sampled_from([torch.int, torch.long]),
        jagged_tensor_dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.int,
                torch.long,
            ]  # Disable torch.bfloat16 due to large error bound
        ),
        has_weights=st.booleans(),
    )
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(40))
    @settings(max_examples=20, deadline=None)
    def test_keyed_jagged_index_select_dim1_int32_overflow(
        self,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        has_weights: bool,
    ) -> None:
        """Test keyed_jagged_index_select_dim1 with num_output_lengths > INT32_MAX.

        Verifies that keyed_jagged_index_select_dim1's forward and backward kernels
        compute correctly when num_output_lengths = num_batches * output_batch_size
        overflows int32.
        Uses num_batches=16,777,216 (2^24) and output_batch_size=128 (2^7)
        so their product = 2^31 = INT32_MAX + 1.

        A small output_batch_size (128) ensures backward has only 128
        gpuAtomicAdd operations per value, matching the regular test and
        avoiding non-deterministic floating-point accumulation errors.

        Only the first batch has length=1 (rest are 0) with
        input_batch_size=1 to minimize output memory. Each non-zero length
        contributes output_batch_size elements to the output tensor, so
        keeping only one non-zero length limits output to 128 elements
        (~512 bytes for float32) while output_offsets and output_lengths
        still have 2.1B entries each to exercise the int64 indexing paths.

        Peak GPU memory (dominated by output_offsets and output_lengths):
            index_dtype=int32: ~24 GB (output_offsets 16 GB + output_lengths 8 GB)
            index_dtype=int64: ~32 GB (output_offsets 16 GB + output_lengths 16 GB)
        This test requires a GPU with at least 40 GB memory.
        """
        num_batches = 16777216
        output_batch_size = 128
        input_batch_size = 1
        device = torch.accelerator.current_accelerator()
        is_float = jagged_tensor_dtype in [
            torch.float,
            torch.half,
            torch.bfloat16,
        ]

        lengths = torch.zeros(num_batches, dtype=index_dtype, device=device)
        lengths[0] = 1
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), lengths.cumsum(0)]
        )
        indices = torch.zeros(output_batch_size, dtype=index_dtype, device=device)

        if is_float:
            values = torch.rand(1, dtype=jagged_tensor_dtype, device=device)
        else:
            values = torch.randint(
                2**16, (1,), dtype=jagged_tensor_dtype, device=device
            )

        if has_weights:
            weights = torch.rand(1, dtype=torch.float, device=device)
        else:
            weights = None

        if is_float:
            values.requires_grad = True

        result = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
            values, lengths, offsets, indices, input_batch_size, weights, None
        )

        output = result[0]
        output_lengths = result[1]

        # Verify output_lengths: batch 0 has length=1 repeated output_batch_size
        # times, all other batches have length=0. Use sum to avoid allocating
        # a 2.1B bool tensor.
        self.assertEqual(output_lengths.numel(), num_batches * output_batch_size)
        self.assertEqual(output_lengths.sum().item(), output_batch_size)

        # Verify output values: output_batch_size copies of values[0]
        self.assertEqual(output.numel(), output_batch_size)
        assert torch.all(output == values.detach()[0])

        if has_weights:
            output_weights = result[2]
            # pyre-ignore[16]
            assert torch.all(output_weights == weights[0])

        # Free output_lengths before backward to reduce peak memory
        del output_lengths, result
        torch.cuda.empty_cache()

        if not is_float:
            return

        # Backward: all output elements come from values[0],
        # so grad_values[0] = sum(grad)
        grad = torch.rand(output_batch_size, dtype=output.dtype, device=device)
        output.backward(grad)

        expected_grad = grad.sum().unsqueeze(0)
        torch.testing.assert_close(
            values.grad,
            expected_grad,
            rtol=1e-2 if jagged_tensor_dtype in [torch.half, torch.bfloat16] else None,
            atol=1e-2 if jagged_tensor_dtype in [torch.half, torch.bfloat16] else None,
        )


if __name__ == "__main__":
    unittest.main()
