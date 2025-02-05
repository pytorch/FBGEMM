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
