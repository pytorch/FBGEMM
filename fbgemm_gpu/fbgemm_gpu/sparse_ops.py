#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


class _StackedJagged2DToDenseFunction(torch.autograd.Function):
    @staticmethod
    # pyre-fixme[14]
    def forward(
        # pyre-fixme[2]
        ctx,
        values: torch.Tensor,
        lengths: torch.Tensor,
        offset_per_key: List[int],
        max_lengths_per_key: List[int],
    ) -> Tuple[torch.Tensor]:
        ctx.B = lengths.size(1)
        ctx.D = values.size(1)
        ctx.total_L = values.size(0)
        ctx.offset_per_key = offset_per_key
        (
            padded_values_per_key,
            offsets_tensor_per_key,
        ) = torch.ops.fbgemm.stacked_jagged_2d_to_dense_forward(
            values,
            lengths,
            offset_per_key,
            max_lengths_per_key,
        )
        ctx.offsets_tensor_per_key = offsets_tensor_per_key
        return tuple(padded_values_per_key)

    @staticmethod
    def backward(
        # pyre-fixme[2]
        ctx,
        *grad_padded_values_per_key: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        B = ctx.B
        D = ctx.D
        total_L = ctx.total_L
        offset_per_key = ctx.offset_per_key
        offsets_tensor_per_key = ctx.offsets_tensor_per_key
        grad_values = torch.ops.fbgemm.stacked_jagged_2d_to_dense_backward(
            B,
            D,
            total_L,
            list(grad_padded_values_per_key),
            offsets_tensor_per_key,
            offset_per_key,
        )
        return grad_values, None, None, None


def jagged_1d_to_dense(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_sequence_length: int,
    padding_value: int,
) -> torch.Tensor:
    return torch.ops.fbgemm.jagged_1d_to_dense(
        values=values,
        offsets=offsets,
        max_sequence_length=max_sequence_length,
        padding_value=padding_value,
    )


def jagged_2d_to_dense(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_sequence_length: int,
) -> torch.Tensor:
    return torch.ops.fbgemm.jagged_2d_to_dense(
        values=values,
        offsets=offsets,
        max_sequence_length=max_sequence_length,
    )


def stacked_jagged_1d_to_dense(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offset_per_key: List[int],
    max_lengths_per_key: List[int],
    padding_value: int,
) -> List[torch.Tensor]:
    return torch.ops.fbgemm.stacked_jagged_1d_to_dense(
        values=values,
        lengths=lengths,
        offset_per_key=offset_per_key,
        max_lengths_per_key=max_lengths_per_key,
        padding_value=padding_value,
    )


def stacked_jagged_2d_to_dense(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offset_per_key: List[int],
    max_lengths_per_key: List[int],
) -> List[torch.Tensor]:
    return list(
        # pyre-fixme[16]
        _StackedJagged2DToDenseFunction.apply(
            values,
            lengths,
            offset_per_key,
            max_lengths_per_key,
        )
    )
