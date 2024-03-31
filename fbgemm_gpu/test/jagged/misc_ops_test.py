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
    from test_utils import cpu_and_maybe_gpu, optests
else:
    from fbgemm_gpu.test.test_utils import cpu_and_maybe_gpu, optests


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class JaggedTensorOpsTest(unittest.TestCase):
    @given(
        batch_size=st.integers(1, 128),
        max_length=st.integers(0, 128),
        max_truncated_length=st.integers(1, 32),
        index_dtype=st.sampled_from([torch.int, torch.long]),
        jagged_tensor_dtype=st.sampled_from(
            [torch.float, torch.half, torch.bfloat16, torch.int, torch.long]
        ),
        use_cpu=st.just(True),
    )
    @settings(max_examples=20, deadline=None)
    def test_jagged_1d_to_truncated_values(
        self,
        max_length: int,
        batch_size: int,
        max_truncated_length: int,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        use_cpu: bool,
    ) -> None:
        device = "cpu" if use_cpu else "cuda"
        is_float = jagged_tensor_dtype in [torch.float, torch.half, torch.bfloat16]
        lengths = torch.randint(
            low=0,
            high=max_length + 1,
            size=(batch_size,),
            dtype=index_dtype,
            device=device,
        )
        n = int(lengths.sum().item())
        if is_float:
            values = torch.rand(
                (n,),
                dtype=jagged_tensor_dtype,
                device=device,
            )
        else:
            values = torch.randint(
                2**16,
                (n,),
                dtype=jagged_tensor_dtype,
                device=device,
            )

        truncated_values = torch.ops.fbgemm.jagged_1d_to_truncated_values(
            values,
            lengths,
            max_truncated_length,
        )
        dense_values = torch.ops.fbgemm.jagged_1d_to_dense(
            values=values,
            offsets=torch.ops.fbgemm.asynchronous_complete_cumsum(lengths),
            max_sequence_length=max_truncated_length,
            padding_value=0,
        )  # [B, N]
        truncated_lengths_ref = torch.clamp(lengths, max=max_truncated_length)
        mask2d = torch.arange(max_truncated_length, device=device).expand(
            batch_size, -1
        ) < truncated_lengths_ref.unsqueeze(-1)
        truncated_values_ref = dense_values[mask2d].view(-1)

        torch.testing.assert_close(truncated_values, truncated_values_ref)

    @given(
        batch_size=st.integers(1, 128),
        max_length=st.integers(0, 128),
        index_dtype=st.sampled_from([torch.int, torch.long]),
        jagged_tensor_dtype=st.sampled_from([torch.int, torch.long]),
        empty_lengths=st.booleans(),
        use_cpu=st.just(True),
    )
    @settings(max_examples=20, deadline=None)
    def test_masked_select_jagged_1d(
        self,
        max_length: int,
        batch_size: int,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        empty_lengths: bool,
        use_cpu: bool,
    ) -> None:
        device = "cpu" if use_cpu else "cuda"
        if empty_lengths:
            lengths = torch.zeros(batch_size, dtype=index_dtype, device=device)
        else:
            lengths = torch.randint(
                low=0,
                high=max_length + 1,
                size=(batch_size,),
                dtype=index_dtype,
                device=device,
            )
        lengths[batch_size // 2] = 0  # test a corner case
        n = int(lengths.sum().item())
        values = torch.randint(
            2**16,
            (n,),
            dtype=jagged_tensor_dtype,
            device=device,
        )
        mask = torch.randint(2, (n,)) > 0

        masked_values, masked_lengths = torch.ops.fbgemm.masked_select_jagged_1d(
            values,
            lengths,
            mask,
        )

        masked_values_ref = values[mask]
        cum_count = torch.cumsum(mask, 0)
        cum_count = torch.cat((cum_count, torch.tensor([0])))
        cum_length = cum_count[torch.cumsum(lengths, 0) - 1]
        cum_length_shift_right = torch.roll(cum_length, 1)
        cum_length_shift_right[0] = 0
        masked_lengths_ref = (cum_length - cum_length_shift_right).to(lengths.dtype)

        torch.testing.assert_close(masked_values, masked_values_ref)
        torch.testing.assert_close(masked_lengths, masked_lengths_ref)

    @given(
        B=st.integers(1, 512),
        max_L=st.integers(1, 1000),
        D=st.integers(1, 32),
        dtype=st.sampled_from([torch.float]),
        device_type=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_softmax(
        self,
        B: int,
        max_L: int,
        D: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)
        torch.backends.cuda.matmul.allow_tf32 = False
        lengths = torch.randint(max_L + 1, size=(B,), device=device)
        total_length = int(lengths.sum().item())
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        values = torch.rand(
            (total_length, D), requires_grad=True, dtype=dtype, device=device
        )
        output, _ = torch.ops.fbgemm.jagged_softmax(
            values,
            offsets,
            max_L,
        )
        values_ref = values.detach().clone().requires_grad_(True)
        output_ref, _ = torch.ops.fbgemm.dense_to_jagged(
            torch.nn.functional.softmax(
                torch.ops.fbgemm.jagged_to_padded_dense(
                    values_ref,
                    [offsets],
                    max_lengths=[max_L],
                    padding_value=-5e7,
                ).transpose(1, 2),
                dim=-1,
            ).permute(0, 2, 1),
            [offsets],
            total_length,
        )

        # verify forward
        torch.testing.assert_close(output, output_ref)

        # verify backward
        grad_output = output.detach().clone().requires_grad_(True)

        output.backward(grad_output)
        output_ref.backward(grad_output)

        torch.testing.assert_close(values.grad, values_ref.grad)


if __name__ == "__main__":
    unittest.main()
