#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random
import unittest
from typing import List, Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import assume, given, settings, Verbosity

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_unavailable
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    from fbgemm_gpu.test.test_utils import gpu_available, gpu_unavailable


def lengths_to_segment_ids(lengths: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(
        torch._dim_arange(lengths, 0).long(),
        lengths.long(),
    )


# Converts lengths + values format to COO format
# [B], [N, D] -> [B, N', D].
# pyre-ignore Missing return annotation [3]
def var_list_to_coo(lengths: torch.Tensor, values: torch.Tensor, N: int, D: int):
    rows = lengths_to_segment_ids(lengths)
    num_rows = lengths.size()[0]
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    output_size = lengths.sum()
    # This does D&H sync
    cols = torch.ops.fbgemm.offsets_range(offsets, output_size)
    indices = torch.stack([rows, cols])
    dims = [num_rows, N, D]
    # torch.sparse_coo_tensor is not supported by torch.fx, wrap it.
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=dims,
    )


class JaggedTensorOpsTest(unittest.TestCase):
    @staticmethod
    def expand_into_jagged_permute_ref_(
        permute: List[int],
        length: List[int],
    ) -> List[int]:
        offsets = [0] + list(itertools.accumulate(length))
        output_permute = []
        for r in permute:
            output_permute.extend(
                range(
                    offsets[r],
                    offsets[r + 1],
                )
            )
        return output_permute

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore [56]
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
            high=1,
            size=(output_batch_size,),
            dtype=index_dtype,
            device="cuda",
        )
        if is_float:
            values = torch.rand(
                int(offsets[-1].item()),
                dtype=jagged_tensor_dtype,
                device="cuda",
            )
        else:
            values = torch.randint(
                2**16,
                (int(offsets[-1].item()),),
                dtype=jagged_tensor_dtype,
                device="cuda",
            )
        values_ref = values.detach().clone()
        if has_weights:
            weights = torch.rand(
                int(offsets[-1].item()),
                dtype=random.choice([torch.float, torch.half]),
                device="cuda",
            )
        else:
            weights = None

        # Only float tensors can require grad
        if is_float:
            values.requires_grad = True
            values_ref.requires_grad = True

        index_select_output = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
            values, lengths, offsets, indices, input_batch_size, weights
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

        grad = torch.rand_like(output)
        grad_ref = grad.detach().clone()

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
