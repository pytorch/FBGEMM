# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import triton
import triton.language as tl


@triton.jit
def dense_jagged_cat_jagged_out_kernel(
    a_ptr,  # dense
    b_ptr,  # jagged
    c_ptr,  # jagged
    b_offsets_ptr,
    c_offsets_ptr,
    max_seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    b_start = tl.load(b_offsets_ptr + pid_batch)
    b_end = tl.load(b_offsets_ptr + pid_batch + 1)
    c_start = b_start + pid_batch
    N = b_end - b_start
    N = tl.minimum(N, max_seq_len)

    a = tl.load(a_ptr + pid_batch)
    tl.store(c_ptr + c_start, a)

    offs_k = tl.arange(0, BLOCK_SIZE)
    for k in range(0, N, BLOCK_SIZE):
        b_offset = k + offs_k
        b_ptrs = b_ptr + b_start + b_offset
        b = tl.load(b_ptrs, mask=b_offset < N, other=0.0)
        tl.store(c_ptr + c_start + 1 + b_offset, b, mask=b_offset < N)
    tl.store(c_offsets_ptr + pid_batch, b_start + pid_batch)


def dense_jagged_cat_jagged_out(
    a: torch.Tensor,
    b: torch.Tensor,
    b_offsets: torch.Tensor,
    max_seq_len: int,
):
    assert a.is_contiguous()
    assert b.is_contiguous()
    assert b_offsets.is_contiguous()
    B = a.size(0)
    BLOCK_SIZE = 128
    c = torch.zeros(b.size(0) + a.size(0), dtype=a.dtype, device=a.device)
    c_offsets = torch.empty(
        b_offsets.size(0), dtype=b_offsets.dtype, device=b_offsets.device
    )  # B + 1

    dense_jagged_cat_jagged_out_kernel[(B,)](
        a,
        b,
        c,
        b_offsets,
        c_offsets,
        max_seq_len,
        # pyre-fixme[6]: For 7th argument expected `constexpr` but got `int`.
        BLOCK_SIZE,
    )

    c_offsets[-1] = b_offsets[-1] + B

    return c, c_offsets
