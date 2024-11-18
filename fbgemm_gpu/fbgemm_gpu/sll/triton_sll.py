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


def set_block_size(N: int) -> int:
    if N > 64:
        return 64
    elif N > 16:
        return 32
    else:
        return 16


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


# TODO add autotune to find best block size
# add supergroup to optimize GPU cache
@triton.jit
def jagged_dense_bmm_kernel(
    a_ptr,
    a_offset_ptr,
    b_ptr,
    c_ptr,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bl,  # batch idx
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    max_seq_len,  # max sequence length for jaggged tensor
    allow_tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (sum_B(M_i), K), B has shape (B, K, N) and C has shape (sum_B(M_i), N)
    """
    pid_batch = tl.program_id(0)
    pid = tl.program_id(1)

    # a_offset_ptr has stride of 1
    # row_start for jagged tensor
    begin = tl.load(a_offset_ptr + pid_batch)
    end = tl.load(a_offset_ptr + pid_batch + 1)
    M = tl.minimum(end - begin, max_seq_len)  # in case M > max seq len
    if M == 0:
        return

    # num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # if pid_m * BLOCK_SIZE_M >=M, then this block doesn't need to be computed
    if pid_m * BLOCK_SIZE_M >= M:
        return

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if pid_n * BLOCK_SIZE_N >= N:
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak + begin * stride_am
    )  # jagged tensor ptr
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
        + pid_batch * stride_bl
    )  # dense tensor ptr

    c = tl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
    )  # TODO, max this flexible

    # Compute c[m, n] for 1 example of the batch
    for k in range(0, K, BLOCK_SIZE_K):
        updated_offset = k + offs_k
        a = tl.load(
            a_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(updated_offset[None, :] < K) & (offs_am[:, None] < M),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(updated_offset[:, None] < K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = (
        c_ptr
        + stride_cm * offs_m[:, None]
        + stride_cn * offs_n[None, :]
        + begin * stride_cm
    )
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def jagged_jagged_bmm_kernel(
    a_ptr,
    a_offset_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cl,
    stride_cm,
    stride_cn,
    max_seq_len,
    allow_tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A has shape (M, sum_B(Ki)), B has shape (sum_B(Ki), N) and C has shape (B, M, N)
    """
    pid_batch = tl.program_id(0)
    pid = tl.program_id(1)

    # need to make sure a_offset_ptr has stride of 1
    begin = tl.load(a_offset_ptr + pid_batch)
    end = tl.load(a_offset_ptr + pid_batch + 1)
    K = end - begin  # K for current pid_batch
    K = tl.minimum(K, max_seq_len)
    # if K == 0:
    #     return

    # calculate pid_m and pid_n
    # num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        a_ptr
        + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        + begin * stride_ak
    )
    b_ptrs = (
        b_ptr
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        + begin * stride_bk
    )

    c = tl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
    )  # TODO, max this flexible
    for k in range(0, K, BLOCK_SIZE_K):
        updated_offset = k + offs_k
        a = tl.load(
            a_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=((updated_offset[None, :] < K) & (offs_am[:, None] < M)),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=((updated_offset[:, None] < K) & (offs_bn[None, :] < N)),
            other=0.0,
        )
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = (
        c_ptr
        + stride_cm * offs_m[:, None]
        + stride_cn * offs_n[None, :]
        + stride_cl * pid_batch
    )

    tl.store(c_ptrs, c, mask=mask)


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


def triton_jagged_dense_bmm(a, b, a_offsets, max_seq_len, allow_tf32):
    # checks constraints
    assert a.shape[1] == b.shape[1], "incompatible dimensions"
    assert a_offsets.is_contiguous(), "A offsets mush be contiguous"
    sum_B, K = a.shape
    B, K, N = b.shape
    # Use zeros instead of empty to handle corner case when jagged tensor has length > max seq len
    # In that case, it is possible that the output is inconsistent with the padded version if empty is used
    c = a.new_zeros((sum_B, N))

    BLOCK_SIZE_M = 32 if max_seq_len < 50 else 64
    BLOCK_SIZE_N = set_block_size(N)
    BLOCK_SIZE_K = set_block_size(K)

    # 2D launch kernel where each block gets its own program.
    # TODO, is this the best way to handle launch grid?
    # The grid number on M axises is larger than required often due to max_seq_len
    grid = (
        B,
        triton.cdiv(max_seq_len, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )

    jagged_dense_bmm_kernel[grid](
        a,
        a_offsets,
        b,
        c,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        max_seq_len,
        allow_tf32,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    return c


def triton_jagged_jagged_bmm(a, b, a_offsets, max_seq_len, allow_tf32):
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a_offsets.is_contiguous(), "A offsets mush be contiguous"
    M, _ = a.shape
    _, N = b.shape
    B = a_offsets.size(0) - 1
    # allocates output
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    # 2D launch kernel where each block gets its own program.
    BLOCK_SIZE_M = set_block_size(M)
    BLOCK_SIZE_N = set_block_size(N)
    BLOCK_SIZE_K = 32
    grid = (
        B,
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )
    jagged_jagged_bmm_kernel[grid](
        a,
        a_offsets,
        b,
        c,
        M,
        N,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        max_seq_len,
        allow_tf32,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    return c


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


class JaggedDenseBmm(torch.autograd.Function):
    """
    Compute batch matrix multiplication between JaggedTensor and dense tensor
    dense: [B, N, D] * [B, D, T] = [B, N, T]
    jagged: [Sum_B, D] * [B, D, T] = [Sum_B, T]
    """

    @staticmethod
    # pyre-fixme
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        x_offsets: torch.Tensor,
        N: int,
        allow_tf32: bool,
    ):
        ctx.save_for_backward(x, y, x_offsets)
        ctx.N = N
        ctx.allow_tf32 = allow_tf32
        return triton_jagged_dense_bmm(x, y, x_offsets, N, allow_tf32=allow_tf32)

    @staticmethod
    # pyre-fixme
    def backward(ctx, grad_output: torch.Tensor):
        """
        # X = [Sum_B, D]
        # Y = [B, D, T]
        # Z = X * Y = [Sum_B, T]
        # dX = dZ * YT # [Sum_B, T] * [B, T, D] = [Sum_B, D]
        # dY = XT * dZ # [D, sum_B] * [sum_B, T] = [D, B, T]
        """

        # logging.info(f"Jagged bmm backward called")

        (x, y, x_offsets) = ctx.saved_tensors
        N = ctx.N
        grad_x = triton_jagged_dense_bmm(
            grad_output, y.permute(0, 2, 1), x_offsets, N, allow_tf32=ctx.allow_tf32
        )
        grad_y = triton_jagged_jagged_bmm(
            x.T, grad_output, x_offsets, N, allow_tf32=ctx.allow_tf32
        )
        return grad_x, grad_y, None, None, None


class JaggedJaggedBmm(torch.autograd.Function):
    """
    Compute batch matrix multiplication between JaggedTensor and Jagged Tensor
    dense: [B, D, N] * [B, N, T] = [B, D, T]
    jagged: [Sum_B, D].T * [Sum_B, T] = [B, D, T]
    """

    @staticmethod
    # pyre-fixme
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        x_offsets: torch.Tensor,
        N: int,
        allow_tf32,
    ):
        ctx.save_for_backward(x, y, x_offsets)
        ctx.N = N
        ctx.allow_tf32 = allow_tf32
        return triton_jagged_jagged_bmm(x.T, y, x_offsets, N, allow_tf32=allow_tf32)

    @staticmethod
    # pyre-fixme
    def backward(ctx, grad_output: torch.Tensor):
        """
        # X = [Sum_B, D]
        # Y = [Sum_B, T]
        # Z = XT * Y = [B, D, T]
        # dXT = dZ * YT -> dX = Y * dZT
        # dY = X * dZ -> X * dZ
        """
        (x, y, offsets) = ctx.saved_tensors
        N = ctx.N
        grad_x = triton_jagged_dense_bmm(
            y, grad_output.permute(0, 2, 1), offsets, N, allow_tf32=ctx.allow_tf32
        )
        grad_y = triton_jagged_dense_bmm(
            x, grad_output, offsets, N, allow_tf32=ctx.allow_tf32
        )
        return grad_x, grad_y, None, None, None


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


def jagged_dense_bmm(
    x: torch.Tensor,
    y: torch.Tensor,
    x_offsets: torch.Tensor,
    N: int,
    allow_tf32: bool,
    use_fbgemm_kernel: bool = True,
) -> torch.Tensor:
    """
    Compute batch matrix multiplication between JaggedTensor and Jagged Tensor
    dense: [B, D, N] * [B, N, T] = [B, D, T]
    jagged: [D, Sum_B] * [Sum_B, T] = [B, D, T]
    """
    if use_fbgemm_kernel:
        return torch.ops.fbgemm.jagged_dense_bmm(x, x_offsets, y, N)[0]
    else:
        return JaggedDenseBmm.apply(x, y, x_offsets, N, allow_tf32)


def jagged_jagged_bmm(
    x: torch.Tensor,
    y: torch.Tensor,
    x_offsets: torch.Tensor,
    N: int,
    allow_tf32: bool,
    use_fbgemm_kernel: bool = True,
):
    """
    Compute batch matrix multiplication between JaggedTensor and Jagged Tensor
    dense: [B, D, N] * [B, N, T] = [B, D, T]
    jagged: [Sum_B, D].T * [Sum_B, T] = [B, D, T]
    """
    if use_fbgemm_kernel:
        return torch.ops.fbgemm.jagged_jagged_bmm(x, y, x_offsets, N)
    else:
        return JaggedJaggedBmm.apply(x, y, x_offsets, N, allow_tf32)


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
    values = values.contiguous()
    offsets = offsets.contiguous()

    return Jagged2ToPaddedDense.apply(values, offsets, max_length, padding_value)
