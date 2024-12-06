# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import torch


class Jagged2SoftmaxMeta(torch.autograd.Function):
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
        y = torch.rand(x.size(0), device=x.device, dtype=x.dtype)

        ctx.save_for_backward(y, x_offsets, row_offsets, head_offsets)
        ctx.max_seq_len_row = max_seq_len_row
        ctx.max_seq_len_head = max_seq_len_head

        return y

    @staticmethod
    # pyre-fixme
    def backward(ctx, grad_output: torch.Tensor):
        y, x_offsets, row_offsets, head_offsets = ctx.saved_tensors
        grad = torch.rand(y.size(), device=y.device, dtype=y.dtype)

        return grad, None, None, None, None, None, None


def meta_jagged2_softmax(
    x: torch.Tensor,
    offsets: torch.Tensor,
    offsets_total: torch.Tensor,
    max_seq_len: int,
    transpose: bool,
) -> torch.Tensor:
    """
    Meta version of jagged2 softmax: [sum(softmax([B_i, B_i]))]
    """
    return Jagged2SoftmaxMeta.apply(
        x,
        offsets_total,
        offsets,
        offsets,
        max_seq_len,
        max_seq_len,
        transpose,
    )


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
    ) -> torch.Tensor:
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

        D = y.size(1)
        L = y.size(0)
        # gradients of the emb vectors beyond max_seq_len is set to zeros
        jagged_C = torch.zeros((L, D), device=y.device, dtype=y.dtype)
        return jagged_C

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

        grad_x = torch.zeros(
            (x.size()), device=grad_output.device, dtype=grad_output.dtype
        )

        # gradients of the emb vectors beyond max_seq_len is set to zeros
        grad_y = torch.zeros(
            y.size(), device=grad_output.device, dtype=grad_output.dtype
        )
        return (
            grad_x,
            grad_y,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# pyre-fixme[3]: Return type must be annotated.
def meta_array_jagged_bmm_jagged_out(
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


class JaggedJaggedBmmNoPaddingMeta(torch.autograd.Function):
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
        assert x.size(1) == y.size(0), "incompatible dimensions"

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

        # pyre-fixme[6]: For 1st argument expected `Sequence[Union[int, SymInt]]`
        #  but got `Tensor`.
        c = torch.rand((z_lengths.sum()), device=x.device, dtype=x.dtype)
        return c

    @staticmethod
    # pyre-fixme
    def backward(ctx, grad_output: torch.Tensor):
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

        grad_x = torch.rand(x.size(), device=x.device, dtype=x.dtype)
        grad_y = torch.rand(y.size(), device=y.device, dtype=y.dtype)
        return grad_x, grad_y, None, None, None, None, None, None, None, None


# pyre-fixme[3]: Return type must be annotated.
def meta_jagged_jagged_bmm_jagged_out(
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
    return JaggedJaggedBmmNoPaddingMeta.apply(
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
