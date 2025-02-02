# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch

from fbgemm_gpu.triton.jagged.triton_jagged_tensor_ops import (
    dense_to_jagged,
    jagged_to_dense,
)


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
