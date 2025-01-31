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


class CPUJaggedDenseElementwiseMul(torch.autograd.Function):
    # NOTE: CPU, GPU, CUDA versions all have their own autograd.Function implementations,
    # ideally we should use one autograd.Function for all of them and do the dispatching
    # inside the autograd.Function.
    """
    Compute elementwise multiplication between jagged tensor and dense tensor.
    z = x * y
    x: [sum_B(L_i)]
    y: dense tensor
    z: [sum_B(L_i)]
    """

    @staticmethod
    def jagged_dense_elementwise_mul_jagged_out(
        jagged: torch.Tensor,
        dense: torch.Tensor,
        seq_lengths: torch.Tensor,
        offsets: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        out = torch.empty_like(jagged)
        for i in range(seq_lengths.size(0)):
            if seq_lengths[i] == 0:
                continue
            a = jagged[offsets[i] : offsets[i + 1]]
            a = a.view(int(seq_lengths[i]), int(seq_lengths[i]))
            out[offsets[i] : offsets[i + 1]] = (
                a * dense[0 : seq_lengths[i], 0 : seq_lengths[i]]
            ).flatten()
        return out

    @staticmethod
    # pyre-fixme
    def forward(
        ctx,  # pyre-ignore [2]
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

        return CPUJaggedDenseElementwiseMul.jagged_dense_elementwise_mul_jagged_out(
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

        grad_x = CPUJaggedDenseElementwiseMul.jagged_dense_elementwise_mul_jagged_out(
            grad_output,
            y,
            x_seq_lengths,
            x_offsets,
            ctx.max_seq_len,
        )

        return grad_x, None, None, None, None


def cpu_jagged_dense_elementwise_mul_jagged_out(
    x: torch.Tensor,
    y: torch.Tensor,
    x_seq_lengths: torch.Tensor,
    x_offsets: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    return CPUJaggedDenseElementwiseMul.apply(
        x,
        y,
        x_seq_lengths,
        x_offsets,
        max_seq_len,
    )


class JaggedSoftmaxCPU(torch.autograd.Function):
    @staticmethod
    # pyre-fixme
    def forward(
        ctx: Any,  # pyre-ignore
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """
        input shpae is [SUM_B, D]
        output shape is [SUM_B, D]
        """
        B = x_offsets.size(0) - 1
        y = torch.zeros(x.size(), device=x.device, dtype=x.dtype)

        for b in range(B):
            y[x_offsets[b] : x_offsets[b + 1], :] = torch.nn.functional.softmax(
                x[x_offsets[b] : x_offsets[b + 1], :], dim=0
            )

        ctx.save_for_backward(y, x_offsets)

        return y

    @staticmethod
    # pyre-fixme
    def backward(
        ctx: Any, grad_output: torch.Tensor  # pyre-ignore
    ) -> Tuple[torch.Tensor, None, None]:
        y, x_offsets = ctx.saved_tensors

        B = x_offsets.size(0) - 1
        grad = torch.zeros(y.size(), device=y.device, dtype=y.dtype)

        for b in range(B):
            curr_y = y[x_offsets[b] : x_offsets[b + 1]]
            curr_grad = grad_output[x_offsets[b] : x_offsets[b + 1]]
            grad[x_offsets[b] : x_offsets[b + 1]] = curr_y * (
                curr_grad - torch.sum(curr_grad * curr_y, dim=0, keepdim=True)
            )

        return grad, None, None


def cpu_jagged_softmax(
    x: torch.Tensor,
    x_offsets: torch.Tensor,
    max_seq_len: int,
    use_fbgemm_kernel: bool = True,
) -> torch.Tensor:
    """
    CPU version of jagged softmax: [sum(softmax([B_i, D]))]
    """
    # Force the CPU backend to use fbgemm kernel as it has better performance
    use_fbgemm_kernel = True
    if use_fbgemm_kernel:
        return torch.ops.fbgemm.jagged_softmax(x, x_offsets, max_seq_len)[0]
    else:
        return JaggedSoftmaxCPU.apply(x, x_offsets, max_seq_len)


class Jagged2SoftmaxCPU(torch.autograd.Function):
    @staticmethod
    # pyre-fixme
    def forward(
        # pyre-fixme[2]: Parameter must be annotated.
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
        y = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        for i in range(B):
            submatrix = x[x_offsets[i] : x_offsets[i + 1]]
            Ni = int(row_offsets[i + 1] - row_offsets[i])
            softmax_dim = 0 if transpose else 1
            y[x_offsets[i] : x_offsets[i + 1]] = torch.nn.functional.softmax(
                submatrix.reshape((Ni, Ni)), dim=softmax_dim
            ).view(-1)

        ctx.save_for_backward(y, x_offsets, row_offsets, head_offsets)
        ctx.max_seq_len_row = max_seq_len_row
        ctx.max_seq_len_head = max_seq_len_head
        ctx.transpose = transpose

        return y

    @staticmethod
    # pyre-fixme
    def backward(ctx, grad_output: torch.Tensor):
        y, x_offsets, row_offsets, head_offsets = ctx.saved_tensors
        B = x_offsets.size(0) - 1
        transpose = ctx.transpose
        softmax_dim = 0 if transpose else -1
        grad = torch.zeros(y.size(0), device=y.device, dtype=y.dtype)

        for i in range(B):
            Ni = row_offsets[i + 1] - row_offsets[i]
            curr_y = y[x_offsets[i] : x_offsets[i + 1]].reshape(Ni, Ni)
            curr_grad = grad_output[x_offsets[i] : x_offsets[i + 1]].reshape(Ni, Ni)
            grad[x_offsets[i] : x_offsets[i + 1]] = (
                curr_y
                * (
                    curr_grad
                    - torch.sum(curr_grad * curr_y, dim=softmax_dim, keepdim=True)
                )
            ).view(-1)

        return grad, None, None, None, None, None, None


def cpu_jagged2_softmax(
    x: torch.Tensor,
    offsets: torch.Tensor,
    offsets_total: torch.Tensor,
    max_seq_len: int,
    transpose: bool,
) -> torch.Tensor:
    """
    GPU version of jagged2 softmax: [sum(softmax([B_i, B_i]))]
    """
    return Jagged2SoftmaxCPU.apply(
        x,
        offsets_total,
        offsets,
        offsets,
        max_seq_len,
        max_seq_len,
        transpose,
    )


# pyre-fixme[3]: Return type must be annotated.
def cpu_jagged_jagged_bmm_jagged_out_kernel(
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_A,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_B,
    # pyre-fixme[2]: Parameter must be annotated.
    max_seq_len,
    # pyre-fixme[2]: Parameter must be annotated.
    lengths_m,
    # pyre-fixme[2]: Parameter must be annotated.
    lengths_n,
    # pyre-fixme[2]: Parameter must be annotated.
    lengths_mn,
    # pyre-fixme[2]: Parameter must be annotated.
    offsets_m,
    # pyre-fixme[2]: Parameter must be annotated.
    offsets_n,
    # pyre-fixme[2]: Parameter must be annotated.
    offsets_mn,
    # pyre-fixme[2]: Parameter must be annotated.
    allow_tf32=False,
):
    jagged_C = torch.empty((int(lengths_mn.sum().item())), dtype=jagged_A.dtype).to(
        jagged_A.device
    )
    B = lengths_m.size(0)

    for i in range(B):
        jagged_C[offsets_mn[i] : offsets_mn[i + 1]] = torch.matmul(
            jagged_A[offsets_m[i] : offsets_m[i + 1]],
            jagged_B[offsets_n[i] : offsets_n[i + 1]].T,
        ).flatten()
    return jagged_C


# pyre-fixme[3]: Return type must be annotated.
def cpu_array_jagged_bmm_jagged_out_kernel(
    # pyre-fixme[2]: Parameter must be annotated.
    array_A,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_B,
    # pyre-fixme[2]: Parameter must be annotated.
    lengths_am,
    # pyre-fixme[2]: Parameter must be annotated.
    lengths_bk,
    # pyre-fixme[2]: Parameter must be annotated.
    lengths_cm,
    # pyre-fixme[2]: Parameter must be annotated.
    offsets_am,
    # pyre-fixme[2]: Parameter must be annotated.
    offsets_bk,
    # pyre-fixme[2]: Parameter must be annotated.
    offsets_cm,
    # pyre-fixme[2]: Parameter must be annotated.
    max_seq_len,
    # pyre-fixme[2]: Parameter must be annotated.
    allow_tf32=False,
    # pyre-fixme[2]: Parameter must be annotated.
    transpose=0,  # one if a is transpose, otherwise zero
):
    B = lengths_am.size(0)
    D = jagged_B.size(1)
    jagged_C = torch.zeros(
        (int(lengths_cm.sum()), D), device=jagged_B.device, dtype=jagged_B.dtype
    )

    for i in range(B):
        seq_len = int(lengths_bk[i])
        capped_seq_len = min(seq_len, max_seq_len)
        a = array_A[offsets_am[i] : offsets_am[i + 1]].view(seq_len, seq_len)
        a = a[:capped_seq_len, :capped_seq_len]

        if transpose:
            a = a.T
        b = jagged_B[offsets_bk[i] : offsets_bk[i] + capped_seq_len]
        jagged_C[offsets_cm[i] : offsets_cm[i] + capped_seq_len] = torch.matmul(a, b)

    return jagged_C


class ArrayJaggedBmmNopaddingCPU(torch.autograd.Function):
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
        # pyre-fixme[2]: Parameter must be annotated.
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
        # pyre-fixme[2]: Parameter must be annotated.
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

        return cpu_array_jagged_bmm_jagged_out_kernel(
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

        grad_x = cpu_jagged_jagged_bmm_jagged_out_kernel(
            grad_output,
            y,
            ctx.max_seq_len,
            z_lengths,
            y_lengths,
            x_lengths,
            z_offsets,
            y_offsets,
            x_offsets,
            ctx.allow_tf32,
        )

        grad_y = cpu_array_jagged_bmm_jagged_out_kernel(
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


# pyre-fixme[3]: Return type must be annotated.
def cpu_array_jagged_bmm_jagged_out(
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
    return ArrayJaggedBmmNopaddingCPU.apply(
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


class JaggedJaggedBmmNoPaddingCPU(torch.autograd.Function):
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
        # pyre-fixme[2]: Parameter must be annotated.
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
        # pyre-fixme[2]: Parameter must be annotated.
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

        return cpu_jagged_jagged_bmm_jagged_out_kernel(
            x,
            y,
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

        grad_x = cpu_array_jagged_bmm_jagged_out_kernel(
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
        grad_y = cpu_array_jagged_bmm_jagged_out_kernel(
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


# pyre-fixme[3]: Return type must be annotated.
def cpu_jagged_jagged_bmm_jagged_out(
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
    return JaggedJaggedBmmNoPaddingCPU.apply(
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


def cpu_jagged_flash_attention_basic(
    q_weights: torch.Tensor,
    k_weights: torch.Tensor,
    v_weights: torch.Tensor,
    offsets: torch.Tensor,
    max_seq_len: int,
    use_mask: bool = False,
    allow_tf32: bool = True,
) -> torch.Tensor:
    num_objects = offsets[1:] - offsets[0:-1:1]
    attn_lengths = num_objects * num_objects
    attn_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(attn_lengths)

    s = torch.ops.fbgemm.sll_jagged_jagged_bmm_jagged_out(
        x=q_weights,
        y=k_weights,  # transpose is done inside the function
        x_lengths=num_objects,
        x_offsets=offsets,
        y_lengths=num_objects,
        y_offsets=offsets,
        z_lengths=attn_lengths,
        z_offsets=attn_offsets,
        max_seq_len=max_seq_len,
        allow_tf32=allow_tf32,
    )

    p = (
        torch.ops.fbgemm.sll_jagged2_softmax(
            x=s,
            offsets=offsets,
            offsets_total=attn_offsets,
            max_seq_len=max_seq_len,
            transpose=False,
        )
        / max_seq_len
    )

    if use_mask:
        attn_mask = torch.triu(
            torch.ones(
                (max_seq_len, max_seq_len),
                dtype=torch.bool,
                device=q_weights.device,
            ),
        ).requires_grad_(False)
        # p = p * attn_mask
        p = torch.ops.fbgemm.sll_jagged_dense_elementwise_mul_jagged_out(
            x=p,
            y=attn_mask,
            x_seq_lengths=num_objects,
            x_offsets=attn_offsets,
            max_seq_len=max_seq_len,
        )

    jagged_O = torch.ops.fbgemm.sll_array_jagged_bmm_jagged_out(
        x=p,
        y=v_weights,
        x_lengths=attn_lengths,
        x_offsets=attn_offsets,
        y_lengths=num_objects,
        y_offsets=offsets,
        z_lengths=num_objects,
        z_offsets=offsets,
        max_seq_len=max_seq_len,
        allow_tf32=allow_tf32,
    )

    return jagged_O


class JaggedDenseAddCPU(torch.autograd.Function):
    @staticmethod
    # pyre-fixme
    def forward(
        ctx: Any,  # pyre-ignore
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        y: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x_offsets)
        ctx.max_seq_len = max_seq_len
        # TODO: what should be the correct behavior when jagged values has length > max seq len?
        # current behavior is to not truncate jagged values
        # similar for backward grad_output
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
            x,
            [x_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )  # [B, max_seq_len, D]
        return torch.ops.fbgemm.dense_to_jagged(padded_x + y, [x_offsets])[0]

    @staticmethod
    # pyre-fixme
    def backward(
        ctx,  # pyre-ignore
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, torch.Tensor, None]:
        (offsets,) = ctx.saved_tensors
        grad_dense = torch.ops.fbgemm.jagged_to_padded_dense(
            grad_output, [offsets], [ctx.max_seq_len]
        )
        return grad_output, None, grad_dense, None


def cpu_jagged_dense_elementwise_add(
    x: torch.Tensor,
    x_offsets: torch.Tensor,
    y: torch.Tensor,
    max_seq_len: int,
    use_fbgemm_kernel: bool = True,
) -> torch.Tensor:
    # Force the CPU backend to use fbgemm kernel as it has better performance
    use_fbgemm_kernel = True
    if use_fbgemm_kernel:
        return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
            x, [x_offsets], y
        )[0]
    else:
        return JaggedDenseAddCPU.apply(x, x_offsets, y, max_seq_len)


def cpu_jagged_dense_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: torch.Tensor,
    offsets: torch.Tensor,
    max_seq_len: int,
    allow_tf32: bool = True,
) -> torch.Tensor:
    """
    q: jagged tensor, [sum_B, D]
    k: dense tensor, [B, D, T]
    v: jagged tensor [sum_B, D]
    attn_bias: dense tensor [B, N, T]
    offsets: offsets for jagged tensor [B + 1]
    """

    # [sum_B, D] * [B, D, T] = [sum_B, T]
    qk = torch.ops.fbgemm.sll_jagged_dense_bmm(
        q,
        k.to(q.dtype),
        offsets,
        max_seq_len,
        allow_tf32=allow_tf32,
        use_fbgemm_kernel=True,
    )

    softmax_input = torch.ops.fbgemm.sll_jagged_dense_elementwise_add(
        qk,
        offsets,
        attn_bias,
        max_seq_len,
        use_fbgemm_kernel=True,
    )

    normed_attn_weights = torch.ops.fbgemm.sll_jagged_softmax(
        softmax_input,
        offsets,
        max_seq_len,
        use_fbgemm_kernel=True,
    )  # [sum_B, T]

    # [sum_B, T] * [sum_B, D] = [B, T, D]
    return torch.ops.fbgemm.sll_jagged_jagged_bmm(
        normed_attn_weights,
        v.to(normed_attn_weights.dtype),
        offsets,
        max_seq_len,
        allow_tf32=allow_tf32,
        use_fbgemm_kernel=True,
    )
