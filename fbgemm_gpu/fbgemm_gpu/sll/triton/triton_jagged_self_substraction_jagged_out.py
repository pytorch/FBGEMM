# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import triton
import triton.language as tl

from .common import next_power_of_two


@triton.jit
def jagged_self_substraction_jagged_out_kernel(
    a_ptr,  # jagged
    b_ptr,  # jagged
    a_offsets_ptr,
    b_offsets_ptr,
    max_seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_index = tl.program_id(1)

    a_offset = tl.load(a_offsets_ptr + pid_batch)
    a_length = tl.load(a_offsets_ptr + pid_batch + 1) - a_offset
    a_length = tl.minimum(a_length, max_seq_len + 1)

    if a_length <= 1:
        return

    N = a_length - 1
    if pid_index >= N:
        return

    a_cur = tl.load(a_ptr + a_offset + pid_index)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a_row = tl.load(a_ptr + a_offset + offs + 1, mask=mask)
    b = a_cur - a_row

    b_offset = tl.load(b_offsets_ptr + pid_batch)
    tl.store(b_ptr + b_offset + pid_index * N + offs, b, mask=mask)


def triton_jagged_self_substraction_jagged_out(
    jagged_A: torch.Tensor,
    offsets_a: torch.Tensor,
    offsets_b: torch.Tensor,
    max_seq_len,
) -> torch.Tensor:
    B = offsets_a.size(0) - 1

    jagged_B = torch.empty(
        (int(offsets_b[-1].item())), device=jagged_A.device, dtype=jagged_A.dtype
    )

    BLOCK_SIZE = max(next_power_of_two(max_seq_len), 16)
    grid = (B, max_seq_len)

    jagged_self_substraction_jagged_out_kernel[grid](
        jagged_A,
        jagged_B,
        offsets_a,
        offsets_b,
        max_seq_len,
        BLOCK_SIZE,
    )

    return jagged_B
