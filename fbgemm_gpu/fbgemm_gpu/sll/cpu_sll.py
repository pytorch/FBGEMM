# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Any, Tuple

import torch


def cpu_jagged_jagged_bmm_kernel(
    x: torch.Tensor, y: torch.Tensor, x_offsets: torch.Tensor, max_seq_len: int
) -> torch.Tensor:
    assert x.size(1) == y.size(0), "incompatible dimensions"
    B = x_offsets.size(0) - 1
    D, _ = x.size()
    _, T = y.size()
    z = torch.empty((B, D, T), dtype=x.dtype, device=x.device)

    for b in range(B):
        z[b, :, :] = torch.mm(
            x[:, x_offsets[b] : x_offsets[b + 1]],
            y[x_offsets[b] : x_offsets[b + 1], :],
        )
    return z


def cpu_jagged_dense_bmm_kernel(
    x: torch.Tensor, y: torch.Tensor, x_offsets: torch.Tensor, max_seq_len: int
) -> torch.Tensor:
    assert x.size(1) == y.size(1), "incompatible dimensions"
    B = x_offsets.size(0) - 1
    z = torch.zeros((x.size(0), y.size(2)), dtype=x.dtype, device=x.device)

    for b in range(B):
        z[x_offsets[b] : x_offsets[b + 1], :] = torch.mm(
            x[x_offsets[b] : x_offsets[b + 1], :], y[b, :, :]
        )
    return z


class JaggedDenseBmmCPU(torch.autograd.Function):
    """
    Compute batch matrix multiplication between JaggedTensor and dense tensor
    dense: [B, N, D] * [B, D, T] = [B, N, T]
    jagged: [Sum_B, D] * [B, D, T] = [Sum_B, T]
    """

    @staticmethod
    # pyre-fixme
    def forward(
        ctx: Any,  # pyre-ignore
        x: torch.Tensor,
        y: torch.Tensor,
        x_offsets: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, y, x_offsets)
        ctx.N = N
        return cpu_jagged_dense_bmm_kernel(x, y, x_offsets, N)

    @staticmethod
    # pyre-fixme
    def backward(
        ctx: Any, grad_output: torch.Tensor  # pyre-ignore
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None, None]:
        """
        # X = [Sum_B, D]
        # Y = [B, D, T]
        # Z = X * Y = [Sum_B, T]
        # dX = dZ * YT # [Sum_B, T] * [B, T, D] = [Sum_B, D]
        # dY = XT * dZ # [D, sum_B] * [sum_B, T] = [D, B, T]
        """
        (x, y, x_offsets) = ctx.saved_tensors
        N = ctx.N
        grad_x = cpu_jagged_dense_bmm_kernel(
            grad_output, y.permute(0, 2, 1), x_offsets, N
        )
        grad_y = cpu_jagged_jagged_bmm_kernel(x.T, grad_output, x_offsets, N)
        return grad_x, grad_y, None, None, None


def cpu_jagged_dense_bmm(
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

    # Force the CPU backend to use fbgemm kernel as it has better performance
    use_fbgemm_kernel = True
    if use_fbgemm_kernel:
        return torch.ops.fbgemm.jagged_dense_bmm(x, x_offsets, y, N)[0]
    else:
        return JaggedDenseBmmCPU.apply(x, y, x_offsets, N)


class JaggedJaggedBmm(torch.autograd.Function):
    """
    Compute batch matrix multiplication between JaggedTensor and Jagged Tensor
    dense: [B, D, N] * [B, N, T] = [B, D, T]
    jagged: [Sum_B, D].T * [Sum_B, T] = [B, D, T]
    """

    @staticmethod
    # pyre-fixme
    def forward(
        ctx: Any,  # pyre-ignore
        x: torch.Tensor,
        y: torch.Tensor,
        x_offsets: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, y, x_offsets)
        ctx.N = N
        return cpu_jagged_jagged_bmm_kernel(x.T, y, x_offsets, N)

    @staticmethod
    # pyre-fixme
    def backward(
        ctx: Any, grad_output: torch.Tensor  # pyre-ignore
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None, None]:
        """
        # X = [Sum_B, D]
        # Y = [Sum_B, T]
        # Z = XT * Y = [B, D, T]
        # dXT = dZ * YT -> dX = Y * dZT
        # dY = X * dZ -> X * dZ
        """
        (x, y, offsets) = ctx.saved_tensors
        N = ctx.N
        grad_x = cpu_jagged_dense_bmm_kernel(
            y, grad_output.permute(0, 2, 1), offsets, N
        )
        grad_y = cpu_jagged_dense_bmm_kernel(x, grad_output, offsets, N)
        return grad_x, grad_y, None, None, None


def cpu_jagged_jagged_bmm(
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
    jagged: [Sum_B, D].T * [Sum_B, T] = [B, D, T]
    """

    # Force the CPU backend to use fbgemm kernel as it has better performance
    use_fbgemm_kernel = True
    if use_fbgemm_kernel:
        return torch.ops.fbgemm.jagged_jagged_bmm(x, y, x_offsets, N)
    else:
        return JaggedJaggedBmm.apply(x, y, x_offsets, N)


def cpu_dense_jagged_cat_jagged_out(
    a: torch.Tensor,
    b: torch.Tensor,
    b_offsets: torch.Tensor,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert a.size(0) == b_offsets.size(0) - 1
    c = torch.empty(b.size(0) + a.size(0), dtype=a.dtype, device=a.device)
    c_offsets = b_offsets + torch.arange(
        b_offsets.size(0), dtype=torch.int64, device=a.device
    )
    lengths = torch.diff(b_offsets)
    c = torch.cat(
        [
            (
                torch.cat((a[i : i + 1], b[b_offsets[i] : b_offsets[i + 1]]), dim=-1)
                if lengths[i] > 0
                else a[i : i + 1]
            )
            for i in range(a.size(0))
        ],
        dim=-1,
    )
    return c, c_offsets


def cpu_jagged_self_substraction_jagged_out(
    jagged_A: torch.Tensor,
    offsets_a: torch.Tensor,
    offsets_b: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    jagged_B = torch.empty(
        (int(offsets_b[-1].item())), device=jagged_A.device, dtype=jagged_A.dtype
    )
    for i in range(len(offsets_a) - 1):
        if offsets_a[i + 1] - offsets_a[i] == 1:
            continue

        a = jagged_A[offsets_a[i] : offsets_a[i + 1]]
        jagged_B[offsets_b[i] : offsets_b[i + 1]] = (
            a[:-1].unsqueeze(1) - a[1:].unsqueeze(0)
        ).flatten()
    return jagged_B


def meta_jagged_self_substraction_jagged_out(
    jagged_A: torch.Tensor,
    offsets_a: torch.Tensor,
    offsets_b: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    return torch.empty(
        [torch.library.get_ctx().new_dynamic_size()],
        dtype=jagged_A.dtype,
        device=jagged_A.device,
    )


def cpu_jagged2_to_padded_dense(
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
    B = offsets.size(0) - 1
    dense_output = torch.full(
        (B, max_length, max_length),
        padding_value,
        dtype=values.dtype,
        device=values.device,
    )
    for b in range(B):
        begin = offsets[b]
        end = offsets[b + 1]
        Ni = int(torch.sqrt(end - begin))
        if Ni == 0:
            continue
        dense_output[b, 0:Ni, 0:Ni] = values[begin:end].view(Ni, Ni)

    return dense_output
