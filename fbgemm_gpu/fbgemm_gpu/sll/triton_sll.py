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


def next_power_of_two(N: int) -> int:
    if N > 4096:
        raise Exception(f"{N} is too large that is not supported yet")

    if N > 2048:
        return 4096
    elif N > 1024:
        return 2048
    elif N > 512:
        return 1024
    elif N > 256:
        return 512
    elif N > 128:
        return 256
    elif N > 64:
        return 128
    elif N > 32:
        return 64
    else:
        return 32


def expect_contiguous(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        return x.contiguous()
    else:
        return x


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


@triton.jit
def jagged_dense_elementwise_mul_jagged_out_kernel(
    a_ptr,  # 1d jagged
    b_ptr,  # dense
    c_ptr,  # 1d jagged
    a_seq_lengths_ptr,
    a_offsets_ptr,
    stride_a,
    stride_bm,
    stride_bn,
    max_seq_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_row_block = tl.program_id(1)

    batch_offset = tl.load(a_offsets_ptr + pid_batch)
    batch_seq_len = tl.load(a_seq_lengths_ptr + pid_batch)
    truncated_seq_len = tl.minimum(batch_seq_len, max_seq_len)

    offs_row = tl.arange(0, BLOCK_M)
    offs_col = tl.arange(0, BLOCK_N)

    rows = pid_row_block * BLOCK_M + offs_row

    # a start + batch offset + row offsets + initial col offsets
    a_ptrs = (
        a_ptr
        + batch_offset * stride_a
        + rows[:, None] * truncated_seq_len
        + offs_col[None, :]
    )

    # b start + row offsets + initial col offsets
    b_ptrs = b_ptr + rows[:, None] * stride_bm + offs_col[None, :] * stride_bn

    # c start + batch offset + row offsets + initial col offsets
    c_ptrs = (
        c_ptr + batch_offset + rows[:, None] * truncated_seq_len + offs_col[None, :]
    )

    for block_start in range(0, truncated_seq_len, BLOCK_N):
        cols = block_start + offs_col
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < truncated_seq_len) & (cols[None, :] < truncated_seq_len)
        a = tl.load(a_ptrs, mask=mask)
        a_ptrs += BLOCK_N

        b = tl.load(b_ptrs, mask=mask)
        b_ptrs += BLOCK_N

        c = a * b
        tl.store(c_ptrs, c, mask=mask)
        c_ptrs += BLOCK_N


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
        BLOCK_SIZE,  # pyre-fixme[6]: For 6th argument expected `constexpr` but got `int`.
    )

    return jagged_B


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


def triton_jagged_dense_elementwise_mul_jagged_out(
    jagged_A,
    dense_B,
    seq_lengths_a,
    offsets_a,
    max_seq_len,
):
    B = seq_lengths_a.size(0)
    total_L = jagged_A.size(0)

    jagged_C = torch.zeros((total_L), device=jagged_A.device, dtype=jagged_A.dtype)

    BLOCK_M = 32
    BLOCK_N = 32
    num_blocks_m = triton.cdiv(max_seq_len, BLOCK_M)
    grid = (B, num_blocks_m)

    jagged_dense_elementwise_mul_jagged_out_kernel[grid](
        jagged_A,
        dense_B,
        jagged_C,
        seq_lengths_a,
        offsets_a,
        jagged_A.stride(0),
        dense_B.stride(0),
        dense_B.stride(1),
        max_seq_len,
        BLOCK_M,
        BLOCK_N,
    )

    return jagged_C


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


class JaggedDenseElementwiseMul(torch.autograd.Function):
    """
    Compute elementwise multiplication between jagged tensor and dense tensor.
    z = x * y
    x: [sum_B(L_i)]
    y: dense tensor
    z: [sum_B(L_i)]
    """

    @staticmethod
    # pyre-fixme
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        x_seq_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
    ):
        ctx.max_seq_len = max_seq_len

        ctx.save_for_backward(
            x,
            y,
            x_seq_lengths,
            x_offsets,
        )

        return triton_jagged_dense_elementwise_mul_jagged_out(
            x,
            y,
            x_seq_lengths,
            x_offsets,
            max_seq_len,
        )

    @staticmethod
    # pyre-fixme
    def backward(ctx, grad_output: torch.Tensor):
        (
            x,
            y,
            x_seq_lengths,
            x_offsets,
        ) = ctx.saved_tensors

        grad_x = triton_jagged_dense_elementwise_mul_jagged_out(
            grad_output,
            y,
            x_seq_lengths,
            x_offsets,
            ctx.max_seq_len,
        )

        return grad_x, None, None, None, None


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


def jagged_dense_elementwise_mul_jagged_out(
    x: torch.Tensor,
    y: torch.Tensor,
    x_seq_lengths: torch.Tensor,
    x_offsets: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    return JaggedDenseElementwiseMul.apply(
        x,
        y,
        x_seq_lengths,
        x_offsets,
        max_seq_len,
    )
