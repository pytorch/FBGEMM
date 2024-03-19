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
    from test_utils import gpu_unavailable, optests
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, optests


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class KeyedJaggedIndexSelectTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
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
        is_float = jagged_tensor_dtype in [torch.float, torch.half, torch.bfloat16]
        lengths = torch.randint(
            low=0,
            high=max_seq_length,
            size=(input_batch_size * num_batches,),
            dtype=index_dtype,
            device="cuda",
        )
        offsets = torch.concat(
            [torch.zeros(1, dtype=torch.long, device="cuda"), lengths.cumsum(0)]
        )
        indices = torch.randint(
            low=0,
            high=input_batch_size,
            size=(output_batch_size,),
            dtype=index_dtype,
            device="cuda",
        )

        # If check_non_contiguous=True, create a tensor that is twice as big
        # and then select only odd indices to make it non contiguous
        values_numel = int(offsets[-1].item())
        values_numel = values_numel * 2 if check_non_contiguous else values_numel

        if is_float:
            values = torch.rand(
                values_numel,
                dtype=jagged_tensor_dtype,
                device="cuda",
            )
        else:
            values = torch.randint(
                2**16,
                (values_numel,),
                dtype=jagged_tensor_dtype,
                device="cuda",
            )
        values_ref = values.detach().clone()

        if check_non_contiguous:
            values = values[1::2]
            values_ref = values_ref[1::2]

        if has_weights:
            weights = torch.rand(
                int(offsets[-1].item()),
                dtype=random.choice([torch.float, torch.half]),
                device="cuda",
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


if __name__ == "__main__":
    unittest.main()
