# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Tuple

import torch
import triton
import triton.language as tl

from .common import expect_contiguous


@triton.jit
def jagged2_to_padded_dense_kernel(
    x_ptr,
    lengths_ptr,
    offsets_ptr,
    output_dense_ptr,
    stride_b,
    stride_m,
    stride_n,
    max_length,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    begin = tl.load(offsets_ptr + pid_batch)
    seqlen = tl.load(lengths_ptr + pid_batch)

    seqlen = tl.minimum(seqlen, max_length)
    if seqlen == 0:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + begin + offs_m[:, None] * seqlen + offs_n[None, :]
    x = tl.load(x_ptrs, mask=((offs_m[:, None] < seqlen) & (offs_n[None, :] < seqlen)))

    out_ptrs = (
        output_dense_ptr
        + pid_batch * stride_b
        + offs_m[:, None] * stride_m
        + offs_n[None, :] * stride_n
    )
    tl.store(
        out_ptrs, x, mask=((offs_m[:, None] < seqlen) & (offs_n[None, :] < seqlen))
    )


@triton.jit
def padded_dense_to_jagged2_kernel(
    x_ptr,
    lengths_ptr,
    offsets_ptr,
    output_jagged_ptr,
    stride_b,
    stride_m,
    stride_n,
    max_length,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    begin = tl.load(offsets_ptr + pid_batch)
    # end = tl.load(offsets_ptr + pid_batch + 1)
    seqlen = tl.load(lengths_ptr + pid_batch)

    seqlen = tl.minimum(seqlen, max_length)

    if seqlen == 0:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = (
        x_ptr
        + pid_batch * stride_b
        + offs_m[:, None] * stride_m
        + offs_n[None, :] * stride_n
    )
    x = tl.load(x_ptrs, mask=((offs_m[:, None] < seqlen) & (offs_n[None, :] < seqlen)))
    out_ptrs = output_jagged_ptr + begin + offs_m[:, None] * seqlen + offs_n[None, :]
    tl.store(
        out_ptrs, x, mask=((offs_m[:, None] < seqlen) & (offs_n[None, :] < seqlen))
    )


def jagged2_to_padded_dense_fwd(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    max_length: int,
    padding_value: float,
) -> torch.Tensor:
    B = offsets.size(0) - 1

    output_dense = torch.full(
        (B, max_length, max_length),
        padding_value,
        dtype=values.dtype,
        device=values.device,
    )
    BLOCK_M = 32
    BLOCK_N = 32
    num_blocks_m = triton.cdiv(max_length, BLOCK_M)
    num_blocks_n = triton.cdiv(max_length, BLOCK_N)
    grid = (num_blocks_m, num_blocks_n, B)

    jagged2_to_padded_dense_kernel[grid](
        values,
        lengths,
        offsets,
        output_dense,
        output_dense.stride(0),
        output_dense.stride(1),
        output_dense.stride(2),
        max_length,
        # pyre-fixme[6]: Incompatible parameter type [6]: expected `constexpr` but got `int`.
        BLOCK_M,
        # pyre-fixme[6]: Incompatible parameter type [6]: expected `constexpr` but got `int`.
        BLOCK_N,
    )

    return output_dense


def padded_dense_to_jagged2_fwd(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    B = values.size(0)
    output_jagged = torch.empty(
        int(offsets[-1]), dtype=values.dtype, device=values.device
    )
    BLOCK_M = 32
    BLOCK_N = 32
    num_blocks_m = triton.cdiv(max_length, BLOCK_M)
    num_blocks_n = triton.cdiv(max_length, BLOCK_N)
    grid = (num_blocks_m, num_blocks_n, B)

    padded_dense_to_jagged2_kernel[grid](
        values,
        lengths,
        offsets,
        output_jagged,
        values.stride(0),
        values.stride(1),
        values.stride(2),
        max_length,
        # pyre-fixme[6]: Incompatible parameter type [6]: expected `constexpr` but got `int`.
        BLOCK_M,
        # pyre-fixme[6]: Incompatible parameter type [6]: expected `constexpr` but got `int`.
        BLOCK_N,
    )

    return output_jagged


class Jagged2ToPaddedDense(torch.autograd.Function):
    @staticmethod
    # pyre-fixme
    def forward(
        ctx,
        values: torch.Tensor,
        offsets: torch.Tensor,
        max_length: int,
        padding_value: float,
    ) -> torch.Tensor:
        lengths_square = offsets[1:] - offsets[0:-1:1]
        lengths = torch.sqrt(lengths_square).to(torch.int32)

        ctx.max_length = max_length
        ctx.save_for_backward(lengths, offsets)

        output = jagged2_to_padded_dense_fwd(
            values, lengths, offsets, max_length, padding_value
        )
        return output

    @staticmethod
    # pyre-fixme
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        max_length = ctx.max_length
        (lengths, offsets) = ctx.saved_tensors
        grad_in = padded_dense_to_jagged2_fwd(grad_output, lengths, offsets, max_length)
        return (grad_in, None, None, None)


def jagged2_to_padded_dense(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_length: int,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    values: jagged tensor with size [sum(Ni * Ni)]
    offsets: offsets for jagged tensor, with size [B + 1]
    max_length: maximum sequence length in the batch
    padding_value: value to use for padding
    return padded dense tensor of size [B, N, N]
    """
    values = expect_contiguous(values)
    offsets = expect_contiguous(offsets)

    return Jagged2ToPaddedDense.apply(values, offsets, max_length, padding_value)
