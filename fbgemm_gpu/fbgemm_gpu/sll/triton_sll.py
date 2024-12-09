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

from fbgemm_gpu.triton.jagged.triton_jagged_tensor_ops import (
    dense_to_jagged,
    jagged_to_dense,
)


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


def expect_contiguous(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        return x.contiguous()
    else:
        return x


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


@triton.jit
def array_jagged_bmm_kernel(
    a_ptr,  # 1D array
    b_ptr,  # jagged matrix
    c_ptr,  # output, jagged matrix
    a_offsets_ptr,
    b_offsets_ptr,
    c_offsets_ptr,
    D,  # emb dimension
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    transpose,  # one if a is transpose, otherwise zero
    max_seq_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    allow_tf32: tl.constexpr,
):

    pid_batch = tl.program_id(2)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    batch_offset_am = tl.load(a_offsets_ptr + pid_batch)
    batch_offset_bk = tl.load(b_offsets_ptr + pid_batch)
    batch_offset_cm = tl.load(c_offsets_ptr + pid_batch)

    # calculate M, N, K
    batch_K = tl.load(b_offsets_ptr + pid_batch + 1) - batch_offset_bk  # b [batch_K, D]
    batch_M = tl.load(c_offsets_ptr + pid_batch + 1) - batch_offset_cm

    # use uncapped seq length to determine strides of a
    stride_am = batch_M * (1 - transpose) + 1 * transpose
    stride_ak = batch_M * transpose + 1 * (1 - transpose)

    # truncate seq length
    batch_K = tl.minimum(batch_K, max_seq_len)
    batch_M = tl.minimum(batch_M, max_seq_len)

    if batch_K == 0:
        return

    batch_N = D

    # c [batch_M, D] boundary check
    if pid_m * BLOCK_SIZE_M >= batch_M or pid_n * BLOCK_SIZE_N >= batch_N:
        return

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % batch_M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % batch_N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        a_ptr
        + batch_offset_am
        + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    )
    b_ptrs = (
        b_ptr
        + batch_offset_bk * stride_bk
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(batch_K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs, mask=offs_k[None, :] < batch_K - k * BLOCK_SIZE_K, other=0.0
        )
        b = tl.load(
            b_ptrs, mask=offs_k[:, None] < batch_K - k * BLOCK_SIZE_K, other=0.0
        )
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr
        + stride_cm * batch_offset_cm
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < batch_M) & (offs_cn[None, :] < batch_N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def jagged_jagged_bmm_jagged_out_kernel(
    a_ptr,
    a_offset_ptr,
    b_ptr,
    b_offset_ptr,
    c_ptr,
    offsets_mn_ptr,
    max_seq_len,
    num_blocks_n,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    allow_tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel for computing C = A x B.
    A has shape (sum_B(Mi), K), B has shape (K, sum_B(Ni))
    and C has shape (sum_B(Mi * Ni))
    """

    pid = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    begin_a = tl.load(a_offset_ptr + pid_batch)
    end_a = tl.load(a_offset_ptr + pid_batch + 1)

    begin_b = tl.load(b_offset_ptr + pid_batch)
    end_b = tl.load(b_offset_ptr + pid_batch + 1)

    offset_mn = tl.load(offsets_mn_ptr + pid_batch)

    M = end_a - begin_a
    M = tl.minimum(M, max_seq_len)

    N = end_b - begin_b
    N = tl.minimum(N, max_seq_len)

    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n

    if pid_m * BLOCK_SIZE_M >= M or pid_n * BLOCK_SIZE_N >= N:
        return

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = (
        a_ptr
        + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        + begin_a * stride_am
    )

    b_ptrs = (
        b_ptr
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        + begin_b * stride_bn
    )

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

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

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offset_mn + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


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


def triton_array_jagged_bmm_jagged_out(
    array_A,
    jagged_B,
    lengths_am,
    lengths_bk,
    lengths_cm,
    offsets_am,
    offsets_bk,
    offsets_cm,
    max_seq_len,
    allow_tf32=False,
    transpose=0,  # one if a is transpose, otherwise zero
):
    B = lengths_am.size(0)
    D = jagged_B.size(1)
    L = jagged_B.size(0)
    # gradients of the emb vectors beyond max_seq_len is set to zeros
    jagged_C = torch.zeros((L, D), device=jagged_B.device, dtype=jagged_B.dtype)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    num_blocks_m = triton.cdiv(max_seq_len, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(D, BLOCK_SIZE_N)
    grid = (num_blocks_n, num_blocks_m, B)

    array_jagged_bmm_kernel[grid](
        array_A,
        jagged_B,
        jagged_C,
        offsets_am,
        offsets_bk,
        offsets_cm,
        D,
        jagged_B.stride(0),
        jagged_B.stride(1),
        jagged_C.stride(0),
        jagged_C.stride(1),
        transpose,
        max_seq_len,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        allow_tf32,
    )

    return jagged_C


def triton_jagged_jagged_bmm_jagged_out(
    jagged_A,
    jagged_B,
    max_seq_len,
    lengths_m,
    lengths_n,
    lengths_mn,
    offsets_m,
    offsets_n,
    offsets_mn,
    allow_tf32=False,
):
    assert jagged_A.size(1) == jagged_B.size(0), "incompatible dimensions"
    assert offsets_mn.is_contiguous(), "mn offsets mush be contiguous"
    assert offsets_m.is_contiguous(), "m offsets mush be contiguous"
    assert offsets_n.is_contiguous(), "n offsets mush be contiguous"

    B = lengths_m.size(0)
    jagged_C = torch.zeros(
        (lengths_mn.sum()), device=jagged_A.device, dtype=jagged_A.dtype
    )

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    num_blocks_m = triton.cdiv(max_seq_len, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(max_seq_len, BLOCK_SIZE_N)
    grid = (num_blocks_m * num_blocks_n, B)

    jagged_jagged_bmm_jagged_out_kernel[grid](
        jagged_A,
        offsets_m,
        jagged_B,
        offsets_n,
        jagged_C,
        offsets_mn,
        max_seq_len,
        num_blocks_n,
        jagged_A.size(1),
        jagged_A.stride(0),
        jagged_A.stride(1),
        jagged_B.stride(0),
        jagged_B.stride(1),
        allow_tf32,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )

    return jagged_C


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


class ArrayJaggedBmmNopadding(torch.autograd.Function):
    """
    Compute batch matrix multiplication between JaggedTensor and JaggedTensor without padding.
    z = X * Y
    x: [Sum_B(N_i, N_i)]
    y: [sum_B(N_i), D]
    z: [sum_B(N_i), D]
    """

    @staticmethod
    # pyre-fixme
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        y_lengths: torch.Tensor,
        y_offsets: torch.Tensor,
        z_lengths: torch.Tensor,
        z_offsets: torch.Tensor,
        max_seq_len: int,
        allow_tf32,
    ):
        ctx.allow_tf32 = allow_tf32
        ctx.max_seq_len = max_seq_len

        ctx.save_for_backward(
            x,
            y,
            x_lengths,
            y_lengths,
            z_lengths,
            x_offsets,
            y_offsets,
            z_offsets,
        )

        return triton_array_jagged_bmm_jagged_out(
            x,
            y,
            x_lengths,
            y_lengths,
            z_lengths,
            x_offsets,
            y_offsets,
            z_offsets,
            max_seq_len,
            allow_tf32,
            0,
        )

    @staticmethod
    # pyre-fixme
    def backward(ctx, grad_output: torch.Tensor):
        """
        z = X * Y
        dX = dZ * YT
        dY = XT * dZ

        dZ: [sum_B(N_i), D]
        YT: [D, sum_B(N_i)] call Y.T
        XT: transposed
        Z: [sum_B(N_i), D]
        """

        (
            x,
            y,
            x_lengths,
            y_lengths,
            z_lengths,
            x_offsets,
            y_offsets,
            z_offsets,
        ) = ctx.saved_tensors

        grad_x = triton_jagged_jagged_bmm_jagged_out(
            grad_output,
            y.T,
            ctx.max_seq_len,
            z_lengths,
            y_lengths,
            x_lengths,
            z_offsets,
            y_offsets,
            x_offsets,
            ctx.allow_tf32,
        )

        grad_y = triton_array_jagged_bmm_jagged_out(
            x,
            grad_output,
            x_lengths,
            y_lengths,
            z_lengths,
            x_offsets,
            y_offsets,
            z_offsets,
            ctx.max_seq_len,
            ctx.allow_tf32,
            1,
        )
        return grad_x, grad_y, None, None, None, None, None, None, None, None


class JaggedJaggedBmmNoPadding(torch.autograd.Function):
    """
    Compute batch matrix multiplication between JaggedTensor and JaggedTensor without padding.
    z = x x y^T
    x: [sum_B(M_i), D]
    y: [sum_B(N_i), D]
    z: [sum_B(M_i * N_i)], assuming M_i = N_i
    """

    @staticmethod
    # pyre-fixme
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        y_lengths: torch.Tensor,
        y_offsets: torch.Tensor,
        z_lengths: torch.Tensor,
        z_offsets: torch.Tensor,
        max_seq_len: int,
        allow_tf32,
    ):
        ctx.allow_tf32 = allow_tf32
        ctx.max_seq_len = max_seq_len

        ctx.save_for_backward(
            x,
            y,
            x_lengths,
            y_lengths,
            z_lengths,
            x_offsets,
            y_offsets,
            z_offsets,
        )

        return triton_jagged_jagged_bmm_jagged_out(
            x,
            y.T,
            max_seq_len,
            x_lengths,
            y_lengths,
            z_lengths,
            x_offsets,
            y_offsets,
            z_offsets,
            allow_tf32,
        )

    @staticmethod
    # pyre-fixme
    def backward(ctx, grad_output: torch.Tensor):
        """
        z = x x y^T
        x: [sum_B(M_i), D]
        y: [sum_B(N_i), D]
        z: [sum_B(M_i * N_i)], assuming M_i = N_i
        dx = dz x (y^T)^T = > dx = dz x y
        d(y^T) = x^T x dz => dy = dz^T x x
        """
        (
            x,
            y,
            x_lengths,
            y_lengths,
            z_lengths,
            x_offsets,
            y_offsets,
            z_offsets,
        ) = ctx.saved_tensors

        grad_x = triton_array_jagged_bmm_jagged_out(
            grad_output,
            y,
            z_lengths,
            y_lengths,
            x_lengths,
            z_offsets,
            y_offsets,
            x_offsets,
            ctx.max_seq_len,
            ctx.allow_tf32,
            transpose=0,
        )
        grad_y = triton_array_jagged_bmm_jagged_out(
            grad_output,
            x,
            z_lengths,
            x_lengths,
            y_lengths,
            z_offsets,
            x_offsets,
            y_offsets,
            ctx.max_seq_len,
            ctx.allow_tf32,
            transpose=1,
        )
        return grad_x, grad_y, None, None, None, None, None, None, None, None


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


def array_jagged_bmm_jagged_out(
    x: torch.Tensor,
    y: torch.Tensor,
    x_lengths: torch.Tensor,
    x_offsets: torch.Tensor,
    y_lengths: torch.Tensor,
    y_offsets: torch.Tensor,
    z_lengths: torch.Tensor,
    z_offsets: torch.Tensor,
    max_seq_len: int,
    allow_tf32: bool = True,
):
    return ArrayJaggedBmmNopadding.apply(
        x,
        y,
        x_lengths,
        x_offsets,
        y_lengths,
        y_offsets,
        z_lengths,
        z_offsets,
        max_seq_len,
        allow_tf32,
    )


def jagged_jagged_bmm_jagged_out(
    x: torch.Tensor,
    y: torch.Tensor,
    x_lengths: torch.Tensor,
    x_offsets: torch.Tensor,
    y_lengths: torch.Tensor,
    y_offsets: torch.Tensor,
    z_lengths: torch.Tensor,
    z_offsets: torch.Tensor,
    max_seq_len: int,
    allow_tf32: bool = True,
):
    return JaggedJaggedBmmNoPadding.apply(
        x,
        y,
        x_lengths,
        x_offsets,
        y_lengths,
        y_offsets,
        z_lengths,
        z_offsets,
        max_seq_len,
        allow_tf32,
    )


@triton.jit
def jagged_flash_attention_basic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    offset_ptr,
    o_ptr,
    lse_i_ptr,
    stride_qm,
    stride_qd,
    stride_kd,
    stride_kn,
    stride_vn,
    stride_vd,
    stride_om,
    stride_od,
    max_seq_len,
    D: tl.constexpr,
    NEXT_D: tl.constexpr,
    use_mask: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    begin = tl.load(offset_ptr + pid_batch)
    end = tl.load(offset_ptr + pid_batch + 1)

    seqlen = end - begin
    seqlen = tl.minimum(seqlen, max_seq_len)

    if pid_m * BLOCK_SIZE_M >= seqlen:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    # Offset till next power of 2 for D
    offs_nextd = tl.arange(0, NEXT_D)

    acc = tl.zeros([BLOCK_SIZE_M, NEXT_D], dtype=tl.float32)

    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    for j in range(0, seqlen, BLOCK_SIZE_N):
        offs_n = tl.arange(0, BLOCK_SIZE_N) + j
        q_ptrs = (
            q_ptr
            + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
            + begin * stride_qm
        )

        k_ptrs = (
            k_ptr
            + (offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn)
            + begin * stride_kn
        )

        qk = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for d in range(0, D, BLOCK_SIZE_D):
            updated_offset = d + offs_d
            q = tl.load(
                q_ptrs,
                # pyre-fixme[16]: `int` has no attribute `__getitem__`.
                mask=((updated_offset[None, :] < D) & (offs_m[:, None] < seqlen)),
                other=0.0,
            )
            k = tl.load(
                k_ptrs,
                mask=((updated_offset[:, None] < D) & (offs_n[None, :] < seqlen)),
                other=0.0,
            )
            qk += tl.dot(q, k, allow_tf32=allow_tf32)

            q_ptrs += BLOCK_SIZE_D * stride_qd
            k_ptrs += BLOCK_SIZE_D * stride_kd

        m_ij = tl.maximum(tl.max(qk, axis=1), m_i)
        # Add the correct mask here
        mn_mask = (offs_m[:, None] < seqlen) & (offs_n[None, :] < seqlen)

        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(mn_mask, p, 0.0)

        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)

        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Load V
        v_ptrs = (
            v_ptr
            + (offs_nextd[None, :] * stride_vd + offs_n[:, None] * stride_vn)
            + begin * stride_vn
        )
        v = tl.load(
            v_ptrs,
            mask=((offs_nextd[None, :] < D) & (offs_n[:, None] < seqlen)),
            other=0.0,
        )

        p /= max_seq_len

        if use_mask:
            attn_mask = offs_m[:, None] - offs_n[None, :]
            attn_mask = tl.where(mn_mask, attn_mask, 0.0)
            attn_mask = tl.where(attn_mask > 0, 0.0, 1.0)
            p = tl.where(attn_mask > 0, p, 0.0)

        p = p.to(v_ptr.dtype.element_ty)
        acc_j = tl.dot(p, v, allow_tf32=allow_tf32)
        acc += acc_j
        m_i = m_ij

    lse_i = m_i + tl.math.log(l_i)
    lse_i_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    lse_i_ptrs = lse_i_ptr + lse_i_offsets + begin

    tl.store(lse_i_ptrs, lse_i, mask=lse_i_offsets < seqlen)

    acc = acc / l_i[:, None]

    # Store O
    o_ptrs = o_ptr + (
        offs_m[:, None] * stride_om
        + offs_nextd[None, :] * stride_od
        + begin * stride_om
    )
    o_mask = (offs_m[:, None] < seqlen) & (offs_nextd[None, :] < D)
    tl.store(o_ptrs, acc, mask=o_mask)


def jagged_flash_attention_basic_fwd(
    jagged_Q,
    jagged_K,
    jagged_V,
    offsets,
    max_seq_len,
    use_mask,
    allow_tf32=False,
):
    assert jagged_Q.size(1) == jagged_K.size(0), "incompatible dimensions"

    B = offsets.size(0) - 1
    D = jagged_Q.size(1)

    jagged_O = torch.zeros_like(jagged_Q)
    lse = torch.empty((jagged_Q.size(0)), device=jagged_Q.device, dtype=jagged_Q.dtype)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = 32

    grid = (triton.cdiv(max_seq_len, BLOCK_SIZE_M), B)

    jagged_flash_attention_basic_kernel[grid](
        jagged_Q,
        jagged_K,
        jagged_V,
        offsets,
        jagged_O,
        lse,
        jagged_Q.stride(0),
        jagged_Q.stride(1),
        jagged_K.stride(0),
        jagged_K.stride(1),
        jagged_V.stride(0),
        jagged_V.stride(1),
        jagged_O.stride(0),
        jagged_O.stride(1),
        max_seq_len,
        D,
        triton.next_power_of_2(D),
        use_mask,
        allow_tf32,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return jagged_O, lse


# Similar to fwd kernel, this one is using a grid of
# (num_blocks_m, B) where num_blocks_m is seq_len / BLOCK_SIZE_M
@triton.jit
def _jagged_flash_attention_bwd_preprocess_basic_kernel(
    o_ptr,
    o_offset_ptr,
    do_ptr,
    delta_ptr,
    stride_om,
    stride_od,
    max_seq_len,
    D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    begin_o = tl.load(o_offset_ptr + pid_batch)
    end_o = tl.load(o_offset_ptr + pid_batch + 1)

    M = end_o - begin_o
    M = tl.minimum(M, max_seq_len)

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_od = tl.arange(0, BLOCK_SIZE_D)

    o_offsets = (
        offs_om[:, None] * stride_om
        + offs_od[None, :] * stride_od
        + begin_o * stride_om
    )
    o_ptrs = o_ptr + o_offsets
    do_ptrs = do_ptr + o_offsets
    o_mask = (offs_om[:, None] < M) & (offs_od[None, :] < D)

    # Load O
    o = tl.load(o_ptrs, mask=o_mask)
    do = tl.load(do_ptrs, mask=o_mask)

    delta = tl.sum(o * do, axis=1)

    tl.store(delta_ptr + begin_o + offs_om, delta, mask=offs_om < M)


@triton.jit
def _jagged_flash_attention_bwd_basic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    offset_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    do_ptr,
    delta_ptr,
    lse_ptr,
    stride_qm,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_om,
    stride_od,
    stride_dqm,
    stride_dqd,
    stride_dkn,
    stride_dkd,
    stride_dvn,
    stride_dvd,
    stride_dom,
    stride_dod,
    max_seq_len,
    D: tl.constexpr,
    use_mask: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_batch = tl.program_id(axis=1)

    begin = tl.load(offset_ptr + pid_batch)
    end = tl.load(offset_ptr + pid_batch + 1)

    M = tl.minimum(end - begin, max_seq_len)

    pid_n = tl.program_id(axis=0)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    q_ptrs = (
        q_ptr
        + begin * stride_qm
        + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    )

    k_ptrs = (
        k_ptr
        + begin * stride_kn
        + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    )

    v_ptrs = (
        v_ptr
        + begin * stride_vn
        + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    )

    do_ptrs = (
        do_ptr
        + begin * stride_dom
        + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
    )

    # Load K and V
    k = tl.load(k_ptrs, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))
    v = tl.load(v_ptrs, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))

    # Initialize dv and dk
    dv = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)

    for begin_m in range(0, M, BLOCK_SIZE_M):
        offs_m_temp = begin_m + offs_m

        # Load Q
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        q = tl.load(q_ptrs, mask=((offs_d[None, :] < D) & (offs_m_temp[:, None] < M)))
        qk = tl.dot(q, tl.trans(k), allow_tf32=allow_tf32)

        mn_mask = (offs_m_temp[:, None] < M) & (offs_n[None, :] < M)

        # Load lse_i
        lse_i = tl.load(lse_ptr + offs_m_temp + begin, mask=offs_m_temp < M)

        p = tl.exp(qk - lse_i[:, None])
        p = tl.where(mn_mask, p, 0.0)
        p /= max_seq_len
        p_masked = p

        attn_mask = None
        if use_mask:
            attn_mask = offs_m_temp[:, None] - offs_n[None, :]
            attn_mask = tl.where(mn_mask, attn_mask, 0.0)
            attn_mask = tl.where(attn_mask > 0, 0.0, 1.0)
            p_masked = tl.where(attn_mask > 0, p, 0.0)

        p_masked = p_masked.to(do_ptr.dtype.element_ty)
        do = tl.load(do_ptrs, mask=((offs_d[None, :] < D) & (offs_m_temp[:, None] < M)))
        dv += tl.dot(tl.trans(p_masked), do, allow_tf32=allow_tf32)
        dp = tl.dot(do, tl.trans(v), allow_tf32=allow_tf32)

        # compute ds = p * (dp - delta[:, None])
        Di = tl.load(delta_ptr + offs_m_temp + begin, mask=offs_m_temp < M)
        dp_masked = dp
        if use_mask:
            dp_masked = tl.where(attn_mask > 0, dp, 0.0)

        ds = p * (dp_masked - Di[:, None] * max_seq_len)

        # compute dk = dot(ds.T, q)
        ds = ds.to(q_ptr.dtype.element_ty)
        dk += tl.dot(tl.trans(ds), q, allow_tf32=allow_tf32)

        q_ptrs += BLOCK_SIZE_M * stride_qm
        do_ptrs += BLOCK_SIZE_M * stride_dom

    # store back dk and dv
    dk_ptrs = (
        dk_ptr
        + begin * stride_dkn
        + (offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd)
    )

    dv_ptrs = (
        dv_ptr
        + begin * stride_dvn
        + (offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd)
    )

    tl.store(dk_ptrs, dk, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))
    tl.store(dv_ptrs, dv, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))

    start_m = tl.program_id(axis=0) * BLOCK_SIZE_N
    offs_m_curr = start_m + tl.arange(0, BLOCK_SIZE_N)

    dq_ptrs_curr = (
        dq_ptr
        + begin * stride_dqm
        + (offs_m_curr[:, None] * stride_dqm + offs_d[None, :] * stride_dqd)
    )

    dq_curr = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)

    q_ptrs_curr = (
        q_ptr
        + begin * stride_qm
        + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    )

    q_curr = tl.load(
        q_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < M))
    )

    # Load lse_i
    lse_i_curr = tl.load(lse_ptr + offs_m_curr + begin, mask=offs_m_curr < M)

    do_ptrs_curr = (
        do_ptr
        + begin * stride_dom
        + (offs_m_curr[:, None] * stride_dom + offs_d[None, :] * stride_dod)
    )

    # Load do
    do_curr = tl.load(
        do_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < M))
    )
    Di_curr = tl.load(delta_ptr + offs_m_curr + begin, mask=offs_m_curr < M)

    # When computing dV, we want to compute [BLOCK_SIZE_N] rows of dV.
    # Therefore, the loop's block size is BLOCK_SIZE_M instead of BLOCK_SIZE_N.
    block_start = 0
    while block_start < M:
        offs_n_curr = block_start + tl.arange(0, BLOCK_SIZE_M)

        k_ptrs_curr = (
            k_ptr
            + begin * stride_kn
            + (offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        )
        v_ptrs_curr = (
            v_ptr
            + begin * stride_vn
            + (offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        )

        k_curr = tl.load(
            k_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_n_curr[:, None] < M))
        )
        v_curr = tl.load(
            v_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_n_curr[:, None] < M))
        )

        qk_curr = tl.dot(q_curr, tl.trans(k_curr), allow_tf32=allow_tf32)
        mn_mask_curr = (offs_m_curr[:, None] < M) & (offs_n_curr[None, :] < M)

        p_curr = tl.exp(qk_curr - lse_i_curr[:, None])
        p_curr = tl.where(mn_mask_curr, p_curr, 0.0)
        p_curr /= max_seq_len

        # compute dp = dot(v, do)
        dp_curr = tl.dot(do_curr, tl.trans(v_curr), allow_tf32=allow_tf32)
        dp_curr_masked = dp_curr

        # compute ds = p * (dp - delta[:, None])
        if use_mask:
            attn_mask = offs_m_curr[:, None] - offs_n_curr[None, :]
            attn_mask = tl.where(mn_mask_curr, attn_mask, 0.0)
            attn_mask = tl.where(attn_mask > 0, 0.0, 1.0)
            dp_curr_masked = tl.where(attn_mask > 0, dp_curr, 0.0)

        ds_curr = p_curr * (dp_curr_masked - Di_curr[:, None] * max_seq_len)

        ds_curr = ds_curr.to(k_ptr.dtype.element_ty)
        dq_curr += tl.dot(ds_curr, k_curr, allow_tf32=allow_tf32)
        block_start += BLOCK_SIZE_M

    tl.store(
        dq_ptrs_curr, dq_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < M))
    )


def jagged_flash_attention_basic_backward(
    jagged_Q,
    # K is non-transposed
    jagged_K,
    jagged_V,
    jagged_O,
    offsets,
    dO,
    lse,
    max_seq_len,
    use_mask,
    allow_tf32=False,
):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32

    B = offsets.size(0) - 1
    num_blocks_m = triton.cdiv(max_seq_len, BLOCK_SIZE_M)
    pre_grid = (num_blocks_m, B)

    BLOCK_SIZE_D = max(triton.next_power_of_2(jagged_Q.size(1)), 16)

    delta = torch.empty_like(lse)
    if not dO.is_contiguous():
        dO = dO.contiguous()

    _jagged_flash_attention_bwd_preprocess_basic_kernel[pre_grid](
        jagged_O,
        offsets,
        dO,
        delta,
        jagged_O.stride(0),
        jagged_O.stride(1),
        max_seq_len,
        jagged_O.size(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_D,
    )

    grid = (triton.cdiv(max_seq_len, BLOCK_SIZE_N), B)

    dq = torch.zeros_like(jagged_Q)
    dk = torch.zeros_like(jagged_K)
    dv = torch.zeros_like(jagged_V)

    D = jagged_Q.size(1)
    _jagged_flash_attention_bwd_basic_kernel[grid](
        jagged_Q,
        jagged_K,
        jagged_V,
        jagged_O,
        offsets,
        dq,
        dk,
        dv,
        dO,
        delta,
        lse,
        jagged_Q.stride(0),
        jagged_Q.stride(1),
        jagged_K.stride(0),
        jagged_K.stride(1),
        jagged_V.stride(0),
        jagged_V.stride(1),
        jagged_O.stride(0),
        jagged_O.stride(1),
        dq.stride(0),
        dq.stride(1),
        dk.stride(0),
        dk.stride(1),
        dv.stride(0),
        dv.stride(1),
        dO.stride(0),
        dO.stride(1),
        max_seq_len,
        D,
        use_mask=use_mask,
        allow_tf32=allow_tf32,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return dq, dk, dv


class JaggedFlashAttentionBasic(torch.autograd.Function):
    @staticmethod
    # pyre-fixme
    def forward(
        ctx,
        jagged_Q: torch.Tensor,
        jagged_K: torch.Tensor,
        jagged_V: torch.Tensor,
        offsets: torch.Tensor,
        max_seq_len: int,
        use_mask: bool = True,
        allow_tf32: bool = False,
    ) -> torch.Tensor:
        ctx.allow_tf32 = allow_tf32
        ctx.max_seq_len = max_seq_len
        ctx.use_mask = use_mask

        jagged_O, lse = jagged_flash_attention_basic_fwd(
            jagged_Q,
            jagged_K.T,
            jagged_V,
            offsets,
            max_seq_len,
            use_mask,
            allow_tf32,
        )

        ctx.save_for_backward(
            jagged_Q,
            jagged_K,
            jagged_V,
            offsets,
            jagged_O,
            lse,
        )

        return jagged_O

    @staticmethod
    # pyre-fixme
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        (
            jagged_Q,
            jagged_K,
            jagged_V,
            offsets,
            jagged_O,
            lse,
        ) = ctx.saved_tensors

        dq, dk, dv = jagged_flash_attention_basic_backward(
            jagged_Q=jagged_Q,
            jagged_K=jagged_K,
            jagged_V=jagged_V,
            jagged_O=jagged_O,
            offsets=offsets,
            dO=grad_output,
            lse=lse,
            max_seq_len=ctx.max_seq_len,
            use_mask=ctx.use_mask,
            allow_tf32=ctx.allow_tf32,
        )

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
        )


def jagged_flash_attention_basic(
    q_weights: torch.Tensor,
    k_weights: torch.Tensor,
    v_weights: torch.Tensor,
    offsets: torch.Tensor,
    max_seq_len: int,
    use_mask: bool = False,
    allow_tf32: bool = True,
) -> torch.Tensor:
    q_weights = expect_contiguous(q_weights)
    k_weights = expect_contiguous(k_weights)
    v_weights = expect_contiguous(v_weights)
    jagged_offsets = expect_contiguous(offsets)

    jagged_O = JaggedFlashAttentionBasic.apply(
        q_weights,
        k_weights,
        v_weights,
        jagged_offsets,
        max_seq_len,
        use_mask,
        allow_tf32,
    )

    return jagged_O


class JaggedDenseAdd(torch.autograd.Function):
    @staticmethod
    # pyre-fixme
    def forward(
        ctx, x: torch.Tensor, x_offsets: torch.Tensor, y: torch.Tensor, max_seq_len: int
    ):
        ctx.save_for_backward(x_offsets)
        ctx.max_seq_len = max_seq_len
        # TODO: what should be the correct behavior when jagged values has length > max seq len?
        # current behavior is to not truncate jagged values
        # similar for backward grad_output
        return dense_to_jagged(
            y, [x_offsets], operation_function="add", operation_jagged_values=x
        )[0]

    @staticmethod
    # pyre-fixme
    def backward(ctx, grad_output: torch.Tensor):
        (offsets,) = ctx.saved_tensors
        grad_dense = jagged_to_dense(grad_output, [offsets], [ctx.max_seq_len])
        return grad_output, None, grad_dense, None


def jagged_dense_elementwise_add(
    x: torch.Tensor,
    x_offsets: torch.Tensor,
    y: torch.Tensor,
    max_seq_len: int,
    use_fbgemm_kernel: bool = True,
):
    if use_fbgemm_kernel:
        return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
            x, [x_offsets], y
        )[0]
    else:
        return JaggedDenseAdd.apply(x, x_offsets, y, max_seq_len)


@triton.jit
def jagged_dense_flash_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ab_ptr,  # attn bias ptr
    o_ptr,
    lse_ptr,
    jagged_offsets_ptr,
    max_seq_len,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kd,
    stride_kt,
    stride_vn,
    stride_vd,
    stride_ab_b,  # attn bias stride batch
    stride_ab_n,
    stride_ab_t,
    stride_ob,
    stride_ot,
    stride_od,
    D: tl.constexpr,
    T: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_batch = tl.program_id(1)

    # begin offset of the current sample
    begin = tl.load(jagged_offsets_ptr + pid_batch)
    # end offset of the current sample
    end = tl.load(jagged_offsets_ptr + pid_batch + 1)

    # The seq length of the current sample
    length = end - begin
    length = tl.minimum(length, max_seq_len)

    if length == 0:
        return

    q_start_ptr = q_ptr + begin * stride_ql
    k_start_ptr = k_ptr + pid_batch * stride_kb
    ab_start_ptr = ab_ptr + pid_batch * stride_ab_b
    v_start_ptr = v_ptr + begin * stride_vn

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    # Load a block of Q into [BLOCK_D, BLOCK_T]
    ki_ptrs = k_start_ptr + offs_d[:, None] * stride_kd + offs_t[None, :] * stride_kt

    ki = tl.load(
        ki_ptrs,
        mask=((offs_d[:, None] < D) & (offs_t[None, :] < T)),
        other=0.0,
    )

    mi = tl.zeros([BLOCK_T], dtype=tl.float32) - float("inf")
    li = tl.zeros([BLOCK_T], dtype=tl.float32)
    oi = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    # Loop through the seq length dimension
    for start_l in range(0, length, BLOCK_L):
        offs_l = start_l + tl.arange(0, BLOCK_L)

        # Load a block of K into [BLOCK_L, BLOCK_D]
        qj_ptrs = (
            q_start_ptr
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            + offs_l[:, None] * stride_ql
            + offs_d[None, :] * stride_qd
        )

        qj = tl.load(
            qj_ptrs,
            mask=((offs_l[:, None] < length) & (offs_d[None, :] < D)),
            other=0.0,
        )

        # gemm [BLOCK_L, BLOCK_D] x [BLOCK_D, BLOCK_T] = [BLOCK_L, BLOCK_T]
        qk = tl.dot(qj, ki, allow_tf32=allow_tf32)

        # Load a block of attn bias into [BLOCK_L, BLOCK_T]
        ab_ptrs = (
            ab_start_ptr + offs_l[:, None] * stride_ab_n + offs_t[None, :] * stride_ab_t
        )

        abij = tl.load(
            ab_ptrs,
            mask=((offs_l[:, None] < length) & (offs_t[None, :] < T)),
            other=0.0,
        )

        # q*k output + attn bias
        qk = qk + abij

        # Note: softmax on axis 0
        mij_hat = tl.max(qk, axis=0)
        mi_new = tl.maximum(mi, mij_hat)
        pij_hat = tl.exp(qk - mi_new[None, :])
        pij_hat = tl.where(
            (offs_l[:, None] < length) & (offs_t[None, :] < T), pij_hat, 0.0
        )
        lij_hat = tl.sum(pij_hat, axis=0)
        alpha = tl.exp(mi - mi_new)
        li_new = alpha * li + lij_hat
        oi = alpha[:, None] * oi

        # Load a block of V into [BLOCK_L, BLOCK_D]
        vj_ptrs = (
            v_start_ptr + offs_l[:, None] * stride_vn + offs_d[None, :] * stride_vd
        )

        vj = tl.load(
            vj_ptrs,
            mask=((offs_l[:, None] < length) & (offs_d[None, :] < D)),
            other=0.0,
        )

        pij_hat = pij_hat.to(v_ptr.dtype.element_ty)
        # gemm [BLOCK_T, BLOCK_L] x [BLOCK_L, BLOCK_D] = [BLOCK_T, BLOCK_D]
        oi = oi + tl.dot(tl.trans(pij_hat), vj, allow_tf32=allow_tf32)

        mi = mi_new
        li = li_new

    oi = oi / li[:, None]

    lse_ptrs = lse_ptr + pid_batch * T + offs_t
    # Save both mi and li to avoid recomputation in backward
    lse_i = mi + tl.log(li)
    tl.store(lse_ptrs, lse_i, mask=(offs_t < T))

    # Write the output [BLOCK_T, BLOCK_D]
    attn_out_ptrs = (
        o_ptr
        + pid_batch * stride_ob
        + offs_t[:, None] * stride_ot
        + offs_d[None, :] * stride_od
    )
    tl.store(attn_out_ptrs, oi, mask=((offs_t[:, None] < T) & (offs_d[None, :] < D)))


def jagged_dense_flash_attention_fwd(
    Q,
    K,
    V,
    attn_bias,
    jagged_offsets,
    max_seq_len,
    allow_tf32=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Q: jagged tensor, [sum_B, D]
    K: dense tensor, [B, D, T]
    V: jagged tensor [sum_B, D]
    attn_bias: dense tensor [B, N, T]
    out: dense tenros: [B, T, D]

    Attention steps:
    1. Q * K: [sum_B, D] * [B, D, T] = [sum_B, T]
    2. softmax_input = Q * K + attn_bias
    [sum_B, T] + [B, N, T] = [sum_B, T]
    3. softmax_out = softmax(softmax_input):
    softmax([sum_B, T]) = [sum_B, T]
    4. softmax_out * V:
    [sum_B, T] * [sum_B, D] = [B, T, D]
    """
    assert Q.size(1) == K.size(1), "incompatible dimensions for Q and K"
    assert Q.size() == V.size(), "incompatible dimensions for Q and V"
    assert jagged_offsets.is_contiguous(), "jagged_offsets must be contiguous"

    (B, D, T) = K.size()
    assert D > 0 and (D & (D - 1)) == 0, "D needs to be a power of two"

    attn_out = torch.zeros(B, T, D, dtype=Q.dtype, device=Q.device)
    lse = torch.empty((B, T), dtype=K.dtype, device=K.device)

    BLOCK_T = 32
    BLOCK_L = 32
    BLOCK_D = D

    num_blocks_t = triton.cdiv(T, BLOCK_T)
    grid = (num_blocks_t, B)

    jagged_dense_flash_attention_fwd_kernel[grid](
        Q,
        K,
        V,
        attn_bias,
        attn_out,
        lse,
        jagged_offsets,
        max_seq_len,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        attn_bias.stride(0),
        attn_bias.stride(1),
        attn_bias.stride(2),
        attn_out.stride(0),
        attn_out.stride(1),
        attn_out.stride(2),
        D,
        T,
        allow_tf32,
        BLOCK_T,  # pyre-ignore
        BLOCK_L,  # pyre-ignore
        BLOCK_D,
    )

    return attn_out, lse


@triton.jit
def _bwd_preprocess_do_o_dot(
    o_ptr,
    do_ptr,
    delta_ptr,
    T,
    stride_ob,
    stride_ot,
    stride_od,
    stride_do_b,
    stride_do_t,
    stride_do_d,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_t = tl.program_id(0)
    offs_t = start_t * BLOCK_T + tl.arange(0, BLOCK_T)
    pid_b = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)

    o_ptrs = (
        o_ptr
        + pid_b * stride_ob
        + offs_t[:, None] * stride_ot
        + offs_d[None, :] * stride_od
    )
    do_ptrs = (
        do_ptr
        + pid_b * stride_do_b
        + offs_t[:, None] * stride_do_t
        + offs_d[None, :] * stride_do_d
    )
    o = tl.load(o_ptrs, mask=(offs_t[:, None] < T), other=0.0)
    do = tl.load(do_ptrs, mask=(offs_t[:, None] < T), other=0.0)
    delta = tl.sum(o * do, axis=1)

    delta_ptrs = delta_ptr + pid_b * T + offs_t
    tl.store(delta_ptrs, delta, mask=(offs_t < T))


@triton.jit
def _jagged_dense_flash_attention_bwd_dv_db_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ab_ptr,  # attn bias
    jagged_offsets_ptr,
    out_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dbias_ptr,
    max_seq_len,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kd,
    stride_kt,
    stride_vl,
    stride_vd,
    stride_ab_b,  # attn bias stride batch
    stride_ab_l,
    stride_ab_t,
    stride_ob,
    stride_ot,
    stride_od,
    stride_dq_l,
    stride_dq_d,
    stride_dv_l,
    stride_dv_d,
    stride_db_b,
    stride_db_l,
    stride_db_t,
    stride_do_b,
    stride_do_t,
    stride_do_d,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)
    # begin offset of the current sample
    begin = tl.load(jagged_offsets_ptr + pid_b)
    # end offset of the current sample
    end = tl.load(jagged_offsets_ptr + pid_b + 1)

    # The seq length of the current sample
    seqlen = end - begin
    seqlen = tl.minimum(seqlen, max_seq_len)

    if seqlen == 0:
        return

    q_start_ptr = q_ptr + begin * stride_ql
    k_start_ptr = k_ptr + pid_b * stride_kb
    ab_start_ptr = ab_ptr + pid_b * stride_ab_b
    v_start_ptr = v_ptr + begin * stride_vl
    do_start_ptr = do_ptr + pid_b * stride_do_b
    dq_start_ptr = dq_ptr + begin * stride_dq_l
    dv_start_ptr = dv_ptr + begin * stride_dv_l
    dbias_start_ptr = dbias_ptr + pid_b * stride_db_b
    delta_ptrs = delta_ptr + pid_b * T
    lse_ptrs = lse_ptr + pid_b * T

    start_l = pid_l * BLOCK_L
    offs_l_curr = start_l + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_T)

    q_ptrs = (
        q_start_ptr + offs_l_curr[:, None] * stride_ql + offs_d[None, :] * stride_qd
    )
    k_ptrs = k_start_ptr + offs_d[:, None] * stride_kd + offs_t[None, :] * stride_kt
    v_ptrs = (
        v_start_ptr + offs_l_curr[:, None] * stride_vl + offs_d[None, :] * stride_vd
    )

    do_ptrs = (
        do_start_ptr + offs_t[:, None] * stride_do_t + offs_d[None, :] * stride_do_d
    )

    dq = tl.zeros([BLOCK_L, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_L, BLOCK_D], dtype=tl.float32)

    # Load a block of q into [BLOCK_L, BLOCK_D]
    q = tl.load(
        q_ptrs,
        mask=((offs_l_curr[:, None] < seqlen) & (offs_d[None, :] < BLOCK_D)),
        other=0.0,
    )
    v = tl.load(v_ptrs, mask=(offs_l_curr[:, None] < seqlen), other=0.0)

    # for start_t in range(0, T, BLOCK_T):
    start_t = 0
    while start_t < T:
        offs_t_curr = start_t + tl.arange(0, BLOCK_T)

        # Load a block of k into [BLOCK_D, BLOCK_T]
        k = tl.load(
            k_ptrs,
            mask=((offs_t_curr[None, :] < T) & (offs_d[:, None] < BLOCK_D)),
            other=0.0,
        )
        qk = tl.zeros([BLOCK_L, BLOCK_T], dtype=tl.float32)

        # gemm [BLOCK_L, BLOCK_D] x [BLOCK_D, BLOCK_T] -> [BLOCK_L, BLOCK_T]
        qk += tl.dot(q, k, allow_tf32=allow_tf32)

        ab_ptrs = (
            ab_start_ptr
            + offs_l_curr[:, None] * stride_ab_l
            + offs_t_curr[None, :] * stride_ab_t
        )

        ab = tl.load(
            ab_ptrs,
            mask=((offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)),
            other=0.0,
        )

        # q*k output + attn bias
        qk = qk + ab

        # Mask out invalid positions for softmax
        qk_mask = (offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)
        qk = tl.where(qk_mask, qk, float("-inf"))

        lse_t = tl.load(
            lse_ptrs + offs_t_curr, mask=(offs_t_curr < T), other=float("inf")
        )
        # Perform softmax
        p = tl.exp(qk - lse_t[None, :])
        p = tl.where(qk_mask, p, 0.0)

        # Compute dv
        # Load a block of do into [BLOCK_T, BLOCK_D]
        do = tl.load(do_ptrs, mask=(offs_t_curr[:, None] < T), other=0.0)

        # gemm [BLOCK_L, BLOCK_T] x [BLOCK_T, BLOCK_D] -> [BLOCK_L, BLOCK_D]
        dv += tl.dot(p, do, allow_tf32=allow_tf32)

        # Compute dp
        delta = tl.load(delta_ptrs + offs_t_curr, mask=(offs_t_curr < T))
        dp = tl.zeros([BLOCK_L, BLOCK_T], dtype=tl.float32)

        # gemm [BLOCK_T, BLOCK_D] x [BLOCK_D, BLOCK_L] = [BLOCK_T, BLOCK_L]
        # [BLOCK_T, BLOCK_L]^T -> [BLOCK_L, BLOCK_T]
        dp += tl.trans(tl.dot(do, tl.trans(v), allow_tf32=allow_tf32))

        # Compute ds = p * (dp - delta)
        ds = p * (dp - delta[None, :])

        # Save dbias = ds
        dbias_ptrs = (
            dbias_start_ptr
            + offs_l_curr[:, None] * stride_db_l
            + offs_t_curr[None, :] * stride_db_t
        )
        tl.store(
            dbias_ptrs,
            ds,
            mask=((offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)),
        )

        # Compute dq
        # gemm [BLOCK_L, BLOCK_T] x [BLOCK_T, BLOCK_D] -> [BLOCK_L, BLOCK_D]
        dq += tl.dot(ds, tl.trans(k), allow_tf32=allow_tf32)

        k_ptrs += BLOCK_T * stride_kt
        do_ptrs += BLOCK_T * stride_do_t
        start_t += BLOCK_T

    dq_ptrs = (
        dq_start_ptr
        + offs_l_curr[:, None] * stride_dq_l
        + offs_d[None, :] * stride_dq_d
    )
    dv_ptrs = (
        dv_start_ptr
        + offs_l_curr[:, None] * stride_dv_l
        + offs_d[None, :] * stride_dv_d
    )
    tl.store(dq_ptrs, dq, mask=(offs_l_curr[:, None] < seqlen))
    tl.store(dv_ptrs, dv, mask=(offs_l_curr[:, None] < seqlen))


@triton.jit
def _jagged_dense_flash_attention_bwd_dk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ab_ptr,  # attn bias
    jagged_offsets_ptr,
    out_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dbias_ptr,
    max_seq_len,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kd,
    stride_kt,
    stride_vl,
    stride_vd,
    stride_ab_b,  # attn bias stride batch
    stride_ab_l,
    stride_ab_t,
    stride_ob,
    stride_ot,
    stride_od,
    stride_dk_b,
    stride_dk_d,
    stride_dk_t,
    stride_do_b,
    stride_do_t,
    stride_do_d,
    D,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    # begin offset of the current sample
    begin = tl.load(jagged_offsets_ptr + pid_b)
    # end offset of the current sample
    end = tl.load(jagged_offsets_ptr + pid_b + 1)

    # The seq length of the current sample
    seqlen = end - begin
    seqlen = tl.minimum(seqlen, max_seq_len)

    if seqlen == 0:
        return

    q_start_ptr = q_ptr + begin * stride_ql
    k_start_ptr = k_ptr + pid_b * stride_kb
    ab_start_ptr = ab_ptr + pid_b * stride_ab_b
    v_start_ptr = v_ptr + begin * stride_vl
    do_start_ptr = do_ptr + pid_b * stride_do_b
    dk_start_ptr = dk_ptr + pid_b * stride_dk_b
    delta_ptrs = delta_ptr + pid_b * T
    lse_ptrs = lse_ptr + pid_b * T

    offs_t_curr = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    k_ptrs = (
        k_start_ptr + offs_d[:, None] * stride_kd + offs_t_curr[None, :] * stride_kt
    )

    do_ptrs = (
        do_start_ptr
        + offs_t_curr[:, None] * stride_do_t
        + offs_d[None, :] * stride_do_d
    )

    dk_ptrs = (
        dk_start_ptr
        + offs_d[:, None] * stride_dk_d
        + offs_t_curr[None, :] * stride_dk_t
    )

    dk = tl.zeros([BLOCK_D, BLOCK_T], dtype=tl.float32)

    # Load a block of k into [BLOCK_D, BLOCK_T]
    k = tl.load(
        k_ptrs,
        mask=((offs_t_curr[None, :] < T) & (offs_d[:, None] < BLOCK_D)),
        other=0.0,
    )

    start_l = 0
    while start_l < seqlen:
        offs_l_curr = start_l + tl.arange(0, BLOCK_L)

        # Load a block of q into [BLOCK_L, BLOCK_D]
        q_ptrs = (
            q_start_ptr + offs_l_curr[:, None] * stride_ql + offs_d[None, :] * stride_qd
        )

        q = tl.load(
            q_ptrs,
            mask=(offs_l_curr[:, None] < seqlen),
            other=0.0,
        )

        v_ptrs = (
            v_start_ptr + offs_l_curr[:, None] * stride_vl + offs_d[None, :] * stride_vd
        )

        v = tl.load(v_ptrs, mask=(offs_l_curr[:, None] < seqlen), other=0.0)

        qk = tl.zeros([BLOCK_L, BLOCK_T], dtype=tl.float32)
        # gemm [BLOCK_L, BLOCK_D] x [BLOCK_D, BLOCK_T] -> [BLOCK_L, BLOCK_T]

        qk = tl.dot(q, k, allow_tf32=allow_tf32)
        qk = tl.where(
            (offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T), qk, 0.0
        )

        ab_ptrs = (
            ab_start_ptr
            + offs_l_curr[:, None] * stride_ab_l
            + offs_t_curr[None, :] * stride_ab_t
        )

        ab = tl.load(
            ab_ptrs,
            mask=((offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)),
            other=0.0,
        )

        # q*k output + attn bias
        qk = qk + ab

        # Mask out invalid positions for softmax
        qk_mask = (offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)
        qk = tl.where(qk_mask, qk, float("-inf"))

        lse_t = tl.load(lse_ptrs + offs_t_curr, mask=(offs_t_curr < T))
        # Perform softmax
        p = tl.exp(qk - lse_t[None, :])
        p = tl.where(qk_mask, p, 0.0)

        # Compute dv
        # Load a block of do into [BLOCK_T, BLOCK_D]
        do = tl.load(do_ptrs, mask=(offs_t_curr[:, None] < T), other=0.0)

        # Compute dp
        delta = tl.load(delta_ptrs + offs_t_curr, mask=(offs_t_curr < T))

        # gemm [BLOCK_T, BLOCK_D] x [BLOCK_D, BLOCK_L] = [BLOCK_T, BLOCK_L]
        # [BLOCK_T, BLOCK_L]^T -> [BLOCK_L, BLOCK_T]
        dp = tl.trans(tl.dot(do, tl.trans(v), allow_tf32=allow_tf32))

        # Compute ds = p * (dp - delta)
        ds = p * (dp - delta[None, :])

        # Compute dk
        # gemm [BLOCK_D, BLOCK_L] x [BLOCK_L, BLOCK_T] = [BLOCK_D, BLOCK_T]
        dk += tl.dot(tl.trans(q), ds, allow_tf32=allow_tf32)

        start_l += BLOCK_L

    tl.store(dk_ptrs, dk, mask=(offs_t_curr[None, :] < T))


def jagged_dense_flash_attention_bwd(
    Q,
    K,
    V,
    Out,
    lse,
    do,  # derivative of attn_out
    attn_bias,
    jagged_offsets,
    max_seq_len,
    allow_tf32=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Q: jagged tensor, [sum_B, D]
    K: dense tensor, [B, D, T]
    V: jagged tensor [sum_B, D]
    Out: dense tensor: [B, T, D]
    lse: dense tensor [B, T]
    do: dense tensor [B, T, D]
    attn_bias: dense tensor [B, N, T]
    jagged_offsets: tensor [B + 1]
    """
    assert Q.size(1) == K.size(1), "incompatible dimensions for Q and K"
    assert Q.size() == V.size(), "incompatible dimensions for Q and V"
    assert lse.size(1) == K.size(2), "incompatible dimensions for LSE and K"

    if not do.is_contiguous():
        do = do.contiguous()

    (B, D, T) = K.size()
    BLOCK_T = 32
    BLOCK_L = 32
    BLOCK_D = D
    num_blocks_k = triton.cdiv(T, BLOCK_T)

    dk = torch.zeros_like(K)
    dq = torch.zeros_like(Q)
    dv = torch.zeros_like(V)
    dbias = torch.zeros_like(attn_bias)

    delta = torch.empty_like(lse)
    _bwd_preprocess_do_o_dot[(num_blocks_k, B)](
        Out,
        do,
        delta,
        T,
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        BLOCK_T,  # pyre-ignore
        BLOCK_D,
    )

    num_blocks_l = triton.cdiv(max_seq_len, BLOCK_L)
    _jagged_dense_flash_attention_bwd_dv_db_dq_kernel[(num_blocks_l, B)](
        Q,
        K,
        V,
        attn_bias,
        jagged_offsets,
        Out,
        do,
        lse,
        delta,
        dq,
        dk,
        dv,
        dbias,
        max_seq_len,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        attn_bias.stride(0),
        attn_bias.stride(1),
        attn_bias.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        dq.stride(0),
        dq.stride(1),
        dv.stride(0),
        dv.stride(1),
        dbias.stride(0),
        dbias.stride(1),
        dbias.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        T,
        BLOCK_T,  # pyre-ignore
        BLOCK_L,  # pyre-ignore
        BLOCK_D,
        allow_tf32,
    )

    num_blocks_t = triton.cdiv(T, BLOCK_T)
    _jagged_dense_flash_attention_bwd_dk_kernel[(num_blocks_t, B)](
        Q,
        K,
        V,
        attn_bias,
        jagged_offsets,
        Out,
        do,
        lse,
        delta,
        dq,
        dk,
        dv,
        dbias,
        max_seq_len,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        attn_bias.stride(0),
        attn_bias.stride(1),
        attn_bias.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        D,
        T,
        BLOCK_T,  # pyre-ignore
        BLOCK_L,  # pyre-ignore
        BLOCK_D,
        allow_tf32,
    )

    return dq, dk, dv, dbias


class JaggedDenseFlashAttention(torch.autograd.Function):
    @staticmethod
    # pyre-fixme
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_bias: torch.Tensor,
        jagged_offsets: torch.Tensor,
        max_seq_len: int,
        allow_tf32: bool = False,
    ) -> torch.Tensor:
        attn_out, lse = jagged_dense_flash_attention_fwd(
            Q, K, V, attn_bias, jagged_offsets, max_seq_len, allow_tf32
        )
        ctx.save_for_backward(Q, K, V, attn_bias, jagged_offsets, lse, attn_out)
        ctx.max_seq_len = max_seq_len
        ctx.allow_tf32 = allow_tf32
        return attn_out

    @staticmethod
    # pyre-fixme
    def backward(
        ctx, do: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None
    ]:
        Q, K, V, attn_bias, jagged_offsets, lse, attn_out = ctx.saved_tensors
        max_seq_len = ctx.max_seq_len
        allow_tf32 = ctx.allow_tf32

        dq, dk, dv, dbias = jagged_dense_flash_attention_bwd(
            Q,
            K,
            V,
            attn_out,
            lse,
            do,
            attn_bias,
            jagged_offsets,
            max_seq_len,
            allow_tf32,
        )
        return dq, dk, dv, dbias, None, None, None


def jagged_dense_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: torch.Tensor,
    offsets: torch.Tensor,
    max_seq_len: int,
    allow_tf32: bool = True,
):
    """
    q: jagged tensor, [sum_B, D]
    k: dense tensor, [B, D, T]
    v: jagged tensor [sum_B, D]
    attn_bias: dense tensor [B, N, T]
    offsets: offsets for jagged tensor [B + 1]
    """

    q = expect_contiguous(q)
    k = expect_contiguous(k)
    v = expect_contiguous(v)
    attn_bias = expect_contiguous(attn_bias)
    offsets = expect_contiguous(offsets)

    return JaggedDenseFlashAttention.apply(
        q, k, v, attn_bias, offsets, max_seq_len, allow_tf32
    )
