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
