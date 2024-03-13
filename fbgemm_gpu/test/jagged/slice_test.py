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
from typing import List, Tuple

import hypothesis.strategies as st
import torch
import torch._dynamo
from hypothesis import assume, given, settings, Verbosity

from .common import additional_decorators, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import optests
else:
    from fbgemm_gpu.test.test_utils import optests


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class JaggedSliceTest(unittest.TestCase):
    @given(
        B=st.integers(10, 512),
        N=st.integers(10, 64),
        slice_length=st.integers(0, 64),
        dtype=st.sampled_from([torch.float]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_slice(
        self,
        B: int,
        N: int,
        slice_length: int,
        dtype: torch.dtype,
    ) -> None:
        assume(B != 0)
        device = torch.device("cpu")
        torch.backends.cuda.matmul.allow_tf32 = False
        lengths = torch.randint(N + 1, size=(B,), device=device)
        start_list = [random.randint(0, max(len_ - 1, 0)) for len_ in lengths.tolist()]
        start = torch.tensor(start_list, device=device)

        total_length = int(lengths.sum().item())
        x_values = torch.rand(
            (total_length), requires_grad=True, dtype=dtype, device=device
        )

        output, output_lengths = torch.ops.fbgemm.jagged_slice(
            x_values,
            lengths,
            start,
            slice_length,
        )
        output_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(output_lengths)

        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        x_values_ref = x_values.detach().clone().requires_grad_(True)

        def jagged_slice_ref(
            x_values: torch.Tensor,
            offsets: torch.Tensor,
            start: torch.Tensor,
            slice_length: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            end_offsets_ = slice_length + start + offsets[:-1]
            end_offsets = torch.where(
                end_offsets_ > offsets[1:], offsets[1:], end_offsets_
            )
            start_offsets = start + offsets[:-1]
            indices_to_select: List[torch.Tensor] = []
            for i in range(end_offsets.size(0)):
                indices_to_select.append(
                    torch.arange(start_offsets[i].item(), end_offsets[i].item())
                )
            output_ref = torch.index_select(x_values, 0, torch.cat(indices_to_select))
            new_lengths = end_offsets - start_offsets
            new_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(new_lengths)
            return output_ref, new_offsets

        output_ref, output_offsets_ref = jagged_slice_ref(
            x_values_ref, offsets, start, slice_length
        )

        # verify forward
        torch.testing.assert_close(
            output, output_ref, msg=f"output={output} output_ref={output_ref}"
        )
        torch.testing.assert_close(
            output_offsets,
            output_offsets_ref,
            msg=f"output_off={output_offsets} output_off_ref={output_offsets_ref}",
        )
        # verify backward
        grad_output = output.detach().clone().requires_grad_(True)

        output.backward(grad_output)
        output_ref.backward(grad_output)

        torch.testing.assert_close(
            x_values.grad,
            x_values_ref.grad,
            msg=f"grad={x_values.grad} x_values_ref.grad={x_values_ref.grad}",
        )

    def test_jagged_slice_errors(
        self,
    ) -> None:
        lengths = torch.tensor([1, 2, 3, 4, 5, 6])
        values = torch.tensor([x + y for x in range(6) for y in range(x)])

        with self.assertRaises(RuntimeError):
            torch.ops.fbgemm.jagged_slice(
                values, lengths, torch.tensor([2, 1, 2, 3, 4, 2]), 7
            )

        with self.assertRaises(RuntimeError):
            torch.ops.fbgemm.jagged_slice(
                values, lengths, torch.tensor([-2, 1, 1, 0, 1, 2]), 7
            )


if __name__ == "__main__":
    unittest.main()
