# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .common import add_docs

add_docs(
    torch.ops.fbgemm.jagged_2d_to_dense,
    """
jagged_2d_to_dense(values, x_offsets, max_sequence_length) -> Tensor

Converts a jagged tensor, with a 2D values array into a dense tensor, padding with zeros.

Args:
    values (Tensor): 2D tensor containing the values of the jagged tensor.

    x_offsets (Tensor): 1D tensor containing the starting point of each jagged row in the values tensor.

    max_sequence_length (int): Maximum length of any row in the jagged dimension.

Returns:
    Tensor: The padded dense tensor

Example:
    >>> values = torch.tensor([[1,1],[2,2],[3,3],[4,4]])
    >>> x_offsets = torch.tensor([0, 1, 3])
    >>> torch.ops.fbgemm.jagged_2d_to_dense(values, x_offsets, 3)
    tensor([[[1, 1],
             [0, 0],
             [0, 0]],
            [[2, 2],
             [3, 3],
             [0, 0]]])

""",
)

add_docs(
    torch.ops.fbgemm.jagged_1d_to_dense,
    """
jagged_1d_to_dense(values, offsets, max_sequence_length, padding_value) -> Tensor)

Converts a jagged tensor, with a 1D values array, into a dense tensor, padding with a specified padding value.

Args:
    values (Tensor): 1D tensor containing the values of the jagged tensor.

    offsets (Tensor): 1D tensor containing the starting point of each jagged row in the values tensor.

    max_sequence_length (int): Maximum length of any row in the jagged dimension.

    padding_value (int): Value to set in the empty areas of the dense output, outside of the jagged tensor coverage.

Returns:
    Tensor: the padded dense tensor

Example:
    >>> values = torch.tensor([1,2,3,4])
    >>> offsets = torch.tensor([0, 1, 3])
    >>> torch.ops.fbgemm.jagged_1d_to_dense(values, x_offsets, 3, 0)
    tensor([[1, 0, 0],
            [2, 3, 0]])

""",
)

add_docs(
    torch.ops.fbgemm.dense_to_jagged,
    """
dense_to_jagged(dense, x_offsets, total_L) -> (Tensor, Tensor[])

Converts a dense tensor into a jagged tensor, given the desired offsets of the resulting dense tensor.

Args:
    dense (Tensor): A dense input tensor to be converted

    x_offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension.

    total_L (int, Optional): Total number of values in the resulting jagged tensor.

Returns:
    (Tensor, Tensor[]): Values and offsets of the resulting jagged tensor. Offsets are identital to those that were input.

Example:
    >>> dense = torch.tensor([[[1, 1], [0, 0], [0, 0]], [[2, 2], [3, 3], [0, 0]]])
    >>> x_offsets = torch.tensor([0, 1, 3])
    >>> torch.ops.fbgemm.dense_to_jagged(dense, [x_offsets])
    (tensor([[1, 1],
             [2, 2],
             [3, 3]]), [tensor([0, 1, 3])])

""",
)


add_docs(
    torch.ops.fbgemm.jagged_to_padded_dense,
    """
jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0) -> Tensor

Converts a jagged tensor into a dense tensor, padding with a specified padding value.

Args:
    values (Tensor): Jagged tensor values

    offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension.

    max_lengths (int[]): A list with max_length for each jagged dimension.

    padding_value (float): Value to set in the empty areas of the dense output, outside of the jagged tensor coverage.

Returns:
    Tensor: the padded dense tensor

Example:
    >>> values = torch.tensor([[1,1],[2,2],[3,3],[4,4]])
    >>> offsets = torch.tensor([0, 1, 3])
    >>> torch.ops.fbgemm.jagged_to_padded_dense(values, [offsets], [3], 7)
    tensor([[[1, 1],
             [7, 7],
             [7, 7]],
            [[2, 2],
             [3, 3],
             [7, 7]]])
""",
)


add_docs(
    torch.ops.fbgemm.jagged_dense_elementwise_add,
    """
jagged_dense_elementwise_add(x_values, x_offsets, y) -> Tensor

Adds a jagged tensor to a dense tensor, resulting in dense tensor. Jagged
tensor input will be padded with zeros for the purposes of the addition.

Args:
    x_values (Tensor): Jagged tensor values

    offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension.

    y (Tensor): A dense tensor

Returns:
    Tensor: The sum of jagged input tensor + y

""",
)


add_docs(
    torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output,
    """
jagged_dense_elementwise_add_jagged_output(x_values, x_offsets, y) -> (Tensor, Tensor[])

Adds a jagged tensor to a dense tensor and, resulting in a jagged tensor with the same structure as the input jagged tensor.

Args:
    x_values (Tensor): Jagged tensor values

    x_offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension.

    y (Tensor): A dense tensor

Returns:
    (Tensor, Tensor[]): Values and offsets of the resulting jagged tensor. Offsets are identital to those that were input.

""",
)


add_docs(
    torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output,
    """
jagged_dense_dense_elementwise_add_jagged_output(x_values, x_offsets, y_0, y_1) -> (Tensor, Tensor[])

Adds a jagged tensor to the sum of two dense tensors, resulting in a jagged tensor with the same structure as the input jagged tensor.

Args:
    x_values (Tensor): Jagged tensor values

    x_offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension.

    y_0 (Tensor): A dense tensor

    y_1 (Tensor): A dense tensor

Returns:
    (Tensor, Tensor[]): Values and offsets of the resulting jagged tensor. Offsets are identital to those that were input.

""",
)


add_docs(
    torch.ops.fbgemm.jagged_dense_elementwise_mul,
    """
jagged_dense_elementwise_mul(x_values, x_offsets, y) -> (Tensor, Tensor[])

Elementwise-multiplies a jagged tensor a dense tensor and, resulting in a jagged tensor with the same structure as the input jagged tensor.

Args:
    x_values (Tensor): Jagged tensor values

    x_offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension.

    y (Tensor): A dense tensor

Returns:
    (Tensor, Tensor[]): Values and offsets of the resulting jagged tensor. Offsets are identital to those that were input.

""",
)

add_docs(
    torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul,
    """
batched_dense_vec_jagged_2d_mul(Tensor v, Tensor a_values, Tensor a_offsets) -> Tensor

Batched vector matrix multiplication of a batched dense vector with a jagged tensor, dense vector is in
size (B * H, max_N) and jagged tensor is in size (B, max_N, H * D) where max_N is the maximum size of
jagged dimension. B * H is the batch size and each multiplies is max_N with [max_N, D]

Args:
    v (Tensor): dense vector tensor

    a_values (Tensor): Jagged tensor values

    a_offsets (Tensor []): A list of jagged offset tensors, one for each jagged dimension.

Returns:
    Tensor: output of batch matmul in size (B * H, D)

""",
)

# add_docs(
#    torch.ops.fbgemm.stacked_jagged_1d_to_dense,
#    """Args:
#                {input}
#            Keyword args:
#                {out}""",
# )
#
#
# add_docs(
#    torch.ops.fbgemm.stacked_jagged_2d_to_dense,
#    """Args:
#                {input}
#            Keyword args:
#                {out}""",
# )
