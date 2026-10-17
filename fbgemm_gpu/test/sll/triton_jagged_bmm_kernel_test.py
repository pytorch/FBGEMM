# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl

try:
    import torch_npu  # noqa F401
except ImportError:
    pass


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
    stride_bl,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    max_seq_len,
    allow_tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid = tl.program_id(1)

    begin = tl.load(a_offset_ptr + pid_batch)
    end = tl.load(a_offset_ptr + pid_batch + 1)
    M = tl.minimum(end - begin, max_seq_len)
    if M == 0:
        return

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    if pid_m * BLOCK_SIZE_M >= M:
        return

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if pid_n * BLOCK_SIZE_N >= N:
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak + begin * stride_am
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
        + pid_batch * stride_bl
    )

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        updated_offset = k + offs_k
        a = tl.load(
            a_ptrs,
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


def triton_jagged_dense_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_offsets: torch.Tensor,
    max_seq_len: int,
    allow_tf32: bool,
) -> torch.Tensor:
    assert a.shape[1] == b.shape[1], "incompatible dimensions"
    assert a_offsets.is_contiguous(), "A offsets must be contiguous"
    sum_b, k = a.shape
    batch_size, _, n = b.shape
    c = a.new_zeros((sum_b, n))

    grid = lambda META: (
        batch_size,
        triton.cdiv(max_seq_len, META["BLOCK_SIZE_M"])
        * triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )

    jagged_dense_bmm_kernel[grid](
        a,
        a_offsets,
        b,
        c,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        max_seq_len,
        allow_tf32,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
    )
    return c


def make_normal_fp16(
    shape: tuple[int, ...],
    mean: float,
    std: float,
    device: torch.device,
) -> torch.Tensor:
    return (torch.randn(shape, device=device, dtype=torch.float32) * std + mean).to(
        torch.float16
    )


def make_offsets(lengths: list[int], device: torch.device) -> torch.Tensor:
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=device)
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            lengths_tensor.cumsum(dim=0),
        ],
        dim=0,
    )


def ref_jagged_dense_bmm(
    x: torch.Tensor,
    y: torch.Tensor,
    offsets: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    batch_size = y.size(0)
    padded_x = x.reshape(batch_size, max_seq_len, x.size(1))
    padded_ref = torch.bmm(padded_x, y)
    return padded_ref.reshape(x.size(0), y.size(2))


def main() -> None:
    device = torch.device("npu")

    batch_size = 2
    query_heads = 10
    max_seq_len = 2048
    head_dim = 32
    dense_batch = batch_size * query_heads

    allow_tf32 = False
    int_arg0 = 0
    int_arg1 = 0

    torch.manual_seed(0)
    query = make_normal_fp16(
        (batch_size, query_heads, max_seq_len, head_dim),
        100.0,
        25.0,
        device,
    )
    lengths = [max_seq_len] * dense_batch
    offsets = make_offsets(lengths, device)

    x = query.reshape(dense_batch * max_seq_len, head_dim)
    y = make_normal_fp16(
        (dense_batch, head_dim, head_dim),
        0.0,
        1.0,
        device,
    )

    output = triton_jagged_dense_bmm(
        x,
        y,
        offsets,
        max_seq_len,
        allow_tf32=allow_tf32,
    )
    reference = ref_jagged_dense_bmm(x, y, offsets, max_seq_len)
    output = output.reshape(batch_size, query_heads, max_seq_len, head_dim)
    reference = reference.reshape(batch_size, query_heads, max_seq_len, head_dim)

    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
    print("PASS")
    print(f"query: {tuple(query.shape)} {query.dtype} {query.device}")
    print(f"y: {tuple(y.shape)} {y.dtype} {y.device}")
    print(f"allow_tf32: {allow_tf32}")
    print(f"int args: {int_arg0}, {int_arg1}")
    print(f"output: {tuple(output.shape)} {output.dtype} {output.device}")


if __name__ == "__main__":
    main()
