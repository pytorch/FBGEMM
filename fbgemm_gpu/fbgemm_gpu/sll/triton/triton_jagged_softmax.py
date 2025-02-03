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
def jagged_softmax_kernel(
    input_ptr,
    output_ptr,
    input_offsets_ptr,
    input_row_stride,
    input_head_stride,
    output_row_stride,
    output_head_stride,
    max_seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # BLOCK_SIZE > N (seq len)
):
    """
    input shpae is [SUM_B, H]
    output shape is [SUM_B, H]
    """

    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    row_begin = tl.load(input_offsets_ptr + pid_batch)
    row_end = tl.load(input_offsets_ptr + pid_batch + 1)
    N = tl.minimum(
        max_seq_len, row_end - row_begin
    )  # number of rows to consider softmax
    if N == 0:
        return

    row_start_ptr = input_ptr + row_begin * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = (
        row_start_ptr + col_offsets * input_row_stride + pid_head * input_head_stride
    )
    row = tl.load(input_ptrs, mask=col_offsets < N, other=-float("inf"))
    row_mins_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_mins_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_begin * output_row_stride
    output_ptrs = (
        output_row_start_ptr
        + col_offsets * output_row_stride
        + pid_head * output_head_stride
    )

    tl.store(output_ptrs, softmax_output, mask=col_offsets < N)


def jagged_softmax_(x: torch.Tensor, x_offsets: torch.Tensor, max_seq_len: int):
    sum_B, H = x.shape
    B = x_offsets.size(0) - 1
    BLOCK_SIZE = max(triton.next_power_of_2(max_seq_len), 8)

    y = torch.zeros(
        sum_B, H, device=x.device, dtype=x.dtype
    )  # use zeros instead of empty to ensure the consistent behavior compare to padded version
    jagged_softmax_kernel[(B, H)](
        x,
        y,
        x_offsets,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        # pyre-fixme[6]: Incompatible parameter type [6]: expected `constexpr` but got `int`.
        max_seq_len,
        # pyre-fixme[6]: Incompatible parameter type [6]: expected `constexpr` but got `int`.
        BLOCK_SIZE,
    )

    return y


@triton.jit
def jagged_softmax_backward_kernel(
    grad_output_ptr,
    softmax_output_ptr,
    grad_input_ptr,  # return value
    input_offsets_ptr,
    grad_output_row_stride,
    grad_output_head_stride,
    softmax_output_row_stride,
    softmax_output_head_stride,
    grad_input_row_stride,
    grad_input_head_stride,
    max_seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    grad_output_ptr shpae is [SUM_B, H]
    softmax_output shape is [SUM_B, H]
    grad_input shape is [SUM_B, H]
    """

    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    row_begin = tl.load(input_offsets_ptr + pid_batch)
    row_end = tl.load(input_offsets_ptr + pid_batch + 1)
    N = tl.minimum(
        max_seq_len, row_end - row_begin
    )  # number of rows to consider softmax

    col_offsets = tl.arange(0, BLOCK_SIZE)
    grad_output_ptrs = (
        grad_output_ptr
        + row_begin * grad_output_row_stride
        + col_offsets * grad_output_row_stride
        + pid_head * grad_output_head_stride
    )
    softmax_output_ptrs = (
        softmax_output_ptr
        + row_begin * softmax_output_row_stride
        + col_offsets * softmax_output_row_stride
        + pid_head * softmax_output_head_stride
    )
    grad_output_row = tl.load(grad_output_ptrs, mask=col_offsets < N, other=0.0)
    softmax_output_row = tl.load(softmax_output_ptrs, mask=col_offsets < N, other=0.0)

    sum_value = tl.sum(grad_output_row * softmax_output_row, axis=0)
    grad_input_row = (grad_output_row - sum_value) * softmax_output_row
    grad_input_ptrs = (
        grad_input_ptr
        + row_begin * grad_input_row_stride
        + col_offsets * grad_input_row_stride
        + pid_head * grad_input_head_stride
    )
    tl.store(grad_input_ptrs, grad_input_row, mask=col_offsets < N)


class JaggedSoftmax(torch.autograd.Function):
    @staticmethod
    # pyre-fixme
    def forward(ctx, x: torch.Tensor, x_offsets: torch.Tensor, max_seq_len: int):
        y = jagged_softmax_(x, x_offsets, max_seq_len)
        ctx.save_for_backward(y, x_offsets)
        ctx.max_seq_len = max_seq_len

        return y

    @staticmethod
    # pyre-fixme
    def backward(ctx, grad_output: torch.Tensor):
        y, x_offsets = ctx.saved_tensors
        max_seq_len = ctx.max_seq_len

        sum_B, H = y.shape
        B = x_offsets.size(0) - 1
        BLOCK_SIZE = max(triton.next_power_of_2(max_seq_len), 8)
        grad = torch.zeros(
            sum_B, H, device=y.device, dtype=y.dtype
        )  # use zeros instead of empty to guarantee the behavior

        jagged_softmax_backward_kernel[(B, H)](
            grad_output,
            y,
            grad,
            x_offsets,
            grad_output.stride(0),
            grad_output.stride(1),
            y.stride(0),
            y.stride(1),
            grad.stride(0),
            grad.stride(1),
            max_seq_len,
            # pyre-fixme[6]: Incompatible parameter type [6]: expected `constexpr` but got `int`.
            BLOCK_SIZE,
        )

        return grad, None, None


def jagged_softmax(
    x: torch.Tensor,
    x_offsets: torch.Tensor,
    max_seq_len: int,
    use_fbgemm_kernel: bool = True,
):
    """
    GPU version of jagged softmax: [sum(softmax([B_i, D]))]
    """
    if use_fbgemm_kernel:
        return torch.ops.fbgemm.jagged_softmax(x, x_offsets, max_seq_len)[0]
    else:
        return JaggedSoftmax.apply(x, x_offsets, max_seq_len)


# works now
# we use row offset for softmax calculation
# for now, offsets row == offsets col
@triton.jit
def jagged_2_softmax_kernel(
    input_ptr,
    output_ptr,
    offsets_row_ptr,  # seq
    offsets_col_ptr,  # head
    offsets_overall_ptr,  # offsets for overall matrix = seq_length_i * head_i
    input_stride,
    output_stride,
    transpose,  # one if a is transpose, otherwise zero
    max_seq_len_row,  # max_seq_len for row (seq)
    max_seq_len_col,  # max_seq_len for col (head)
    BLOCK_SIZE: tl.constexpr,  # BLOCK_SIZE > seq_length
):
    """
    input shape is [sum_B(Ni * Hi)]
    output shape is [sum_B(Ni * Hi)]
    Padded version = [B, N, H]
    Calculate softmax alone N dim
    Each kernel calulates softmax for 1 sample and 1 head
    offsets_row.size == offsets_col.size == offsets_overall.size
    """

    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    # start location of current example
    begin = tl.load(offsets_overall_ptr + pid_batch)
    # end = tl.load(offsets_overall_ptr + pid_batch + 1)  # noqa F841
    # end - begin = M_i * N_i

    # softmax on row
    if transpose:
        N = tl.load(offsets_row_ptr + pid_batch + 1) - tl.load(
            offsets_row_ptr + pid_batch
        )
        H = tl.load(offsets_col_ptr + pid_batch + 1) - tl.load(
            offsets_col_ptr + pid_batch
        )
        stride_n = H
        stride_h = H // H  # 1
        # sometimes H is larger than max_seq_len_col
        H = tl.minimum(max_seq_len_col, H)
        N = tl.minimum(max_seq_len_row, N)
    # softmax on col
    else:
        N = tl.load(offsets_col_ptr + pid_batch + 1) - tl.load(
            offsets_col_ptr + pid_batch
        )
        H = tl.load(offsets_row_ptr + pid_batch + 1) - tl.load(
            offsets_row_ptr + pid_batch
        )
        stride_h = N
        stride_n = N // N  # 1
        H = tl.minimum(max_seq_len_row, H)
        N = tl.minimum(max_seq_len_col, N)

    if pid_head >= H:  # TODO double check the equal here
        return
    if H == 0 or N == 0:
        return

    # start of the current example
    start_ptr = input_ptr + begin * input_stride
    # offset for n
    offsets = tl.arange(0, BLOCK_SIZE)

    # Load a softmax row
    input_ptrs = (
        start_ptr
        + offsets * input_stride * stride_n
        + pid_head * input_stride * stride_h
    )  # start + n offsets + head offset
    row = tl.load(input_ptrs, mask=offsets < N, other=-float("inf"))
    row_mins_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_mins_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # calculate output ptr, should be similar to input
    output_start_ptr = output_ptr + begin * output_stride
    output_ptrs = (
        output_start_ptr
        + offsets * output_stride * stride_n
        + pid_head * output_stride * stride_h
    )
    tl.store(output_ptrs, softmax_output, mask=offsets < N)


# TODO, pending test
@triton.jit
def jagged_2_softmax_backward_kernel(
    grad_output_ptr,  # input
    softmax_output_ptr,
    grad_input_ptr,  # return value
    offsets_row_ptr,
    offsets_col_ptr,
    offsets_overall_ptr,
    grad_output_stride,
    softmax_output_stride,
    grad_input_stride,
    transpose,  # transpose
    max_seq_len_row: tl.constexpr,
    max_seq_len_col: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    begin = tl.load(offsets_overall_ptr + pid_batch)
    # end = tl.load(offsets_overall_ptr + pid_batch + 1)  # noqa F841

    # softmax on row
    if transpose:
        N = tl.load(offsets_row_ptr + pid_batch + 1) - tl.load(
            offsets_row_ptr + pid_batch
        )
        H = tl.load(offsets_col_ptr + pid_batch + 1) - tl.load(
            offsets_col_ptr + pid_batch
        )
        stride_n = H
        stride_h = H // H  # 1
        # sometimes H is larger than max_seq_len_col
        H = tl.minimum(max_seq_len_col, H)
        N = tl.minimum(max_seq_len_row, N)
    # softmax on col
    else:
        N = tl.load(offsets_col_ptr + pid_batch + 1) - tl.load(
            offsets_col_ptr + pid_batch
        )
        H = tl.load(offsets_row_ptr + pid_batch + 1) - tl.load(
            offsets_row_ptr + pid_batch
        )
        stride_h = N
        stride_n = N // N  # 1
        H = tl.minimum(max_seq_len_row, H)
        N = tl.minimum(max_seq_len_col, N)

    if pid_head >= H:
        return
    if H == 0 or N == 0:
        pass

    start_ptr = grad_output_ptr + begin * grad_output_stride
    offsets = tl.arange(0, BLOCK_SIZE)

    grad_output_ptrs = (
        start_ptr
        + offsets * grad_output_stride * stride_n
        + pid_head * grad_output_stride * stride_h
    )
    softmax_output_ptrs = (
        softmax_output_ptr
        + begin * softmax_output_stride
        + offsets * softmax_output_stride * stride_n
        + pid_head * softmax_output_stride * stride_h
    )

    grad_output_row = tl.load(grad_output_ptrs, mask=offsets < N, other=0.0)
    softmax_output_row = tl.load(softmax_output_ptrs, mask=offsets < N, other=0.0)

    sum_value = tl.sum(grad_output_row * softmax_output_row, axis=0)
    grad_input_row = (grad_output_row - sum_value) * softmax_output_row

    grad_input_row_start_ptr = grad_input_ptr + begin * grad_input_stride
    grad_input_ptrs = (
        grad_input_row_start_ptr
        + offsets * grad_input_stride * stride_n
        + pid_head * grad_input_stride * stride_h
    )
    tl.store(grad_input_ptrs, grad_input_row, mask=offsets < N)


class Jagged2Softmax(torch.autograd.Function):
    @staticmethod
    # pyre-fixme
    def forward(
        ctx,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        row_offsets: torch.Tensor,
        head_offsets: torch.Tensor,
        max_seq_len_row: int,
        max_seq_len_head: int,
        transpose: bool = True,
    ) -> torch.Tensor:
        B = x_offsets.size(0) - 1
        BLOCK_SIZE = max(triton.next_power_of_2(max_seq_len_row), 8)

        y = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        jagged_2_softmax_kernel[(B, max_seq_len_head)](
            x,
            y,
            row_offsets,
            head_offsets,
            x_offsets,
            x.stride(0),
            y.stride(0),
            transpose,  # transpose
            max_seq_len_row,
            max_seq_len_head,
            # pyre-fixme[6]: Incompatible parameter type [6]: expected `constexpr` but got `int`.
            BLOCK_SIZE,
        )

        ctx.save_for_backward(y, x_offsets, row_offsets, head_offsets)
        ctx.max_seq_len_row = max_seq_len_row
        ctx.max_seq_len_head = max_seq_len_head
        ctx.transpose = transpose

        return y

    @staticmethod
    # pyre-fixme
    def backward(ctx, grad_output: torch.Tensor):
        # TODO: currently backward kernel have small numerical issues.
        y, x_offsets, row_offsets, head_offsets = ctx.saved_tensors
        B = x_offsets.size(0) - 1
        max_seq_len_row = ctx.max_seq_len_row
        max_seq_len_head = ctx.max_seq_len_head
        BLOCK_SIZE = max(triton.next_power_of_2(max_seq_len_row), 8)

        grad = torch.zeros(y.size(0), device=y.device, dtype=y.dtype)

        jagged_2_softmax_backward_kernel[(B, max_seq_len_head)](
            grad_output,
            y,
            grad,
            row_offsets,
            head_offsets,
            x_offsets,
            grad_output.stride(0),
            softmax_output_stride=y.stride(0),
            grad_input_stride=grad.stride(0),
            transpose=ctx.transpose,  # transpose
            max_seq_len_row=max_seq_len_row,
            max_seq_len_col=max_seq_len_head,
            # pyre-fixme[6]: Incompatible parameter type [6]: expected `constexpr` but got `int`.
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad, None, None, None, None, None, None


def jagged2_softmax(
    x: torch.Tensor,
    offsets: torch.Tensor,
    offsets_total: torch.Tensor,
    max_seq_len: int,
    transpose: bool,
):
    """
    GPU version of jagged2 softmax: [sum(softmax([B_i, B_i]))]
    """
    return Jagged2Softmax.apply(
        x,
        offsets_total,
        offsets,
        offsets,
        max_seq_len,
        max_seq_len,
        transpose,
    )
