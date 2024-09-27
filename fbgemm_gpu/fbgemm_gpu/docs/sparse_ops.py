# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .common import add_docs

add_docs(
    torch.ops.fbgemm.permute_2D_sparse_data,
    """
permute_2D_sparse_data(permute, lengths, values, weights=None, permuted_lengths_sum=None) -> Tuple[Tensor, Tensor, Optional[Tensor]]

Permute 2D sparse data along the first dimension (dim 0). Note that 2D
refers to the number of dense dimensions. The input data is actually 3D
where the first two dimensions are dense and the last dimension is
jagged (sparse). The data to permute over can be less or more and with or
without repetitions.

Args:
    permute (Tensor): A 1D-tensor that describes how data is permuted along dim
        0. `permute[i]` indicates that data at position `permute[i]` is moved
        to position `i`. The length of this tensor is the total amount of data
        in dim 0 to be permuted. The values in `permute` must be >= 0 and <
        `lengths.shape[0]`

    lengths (Tensor): A 2D-tensor that contains jagged shapes corresponding to
        the other two dense dimensions. For example, in the case of the
        embedding input, the 3D shape is (num features, batch size, bag size).
        `lengths[t][b]` represents the bag size of feature `t` and sample `b`.

    values (Tensor): A 1D-input-tensor to be permuted. The length of this
        tensor must be equal to `lengths.sum()`. This tensor can be of any data
        type.

    weights (Optional[Tensor] = None): An optional 1D-float-tensor. It must
        have the same length as `values`. It will be permuted the same way as
        values

    permuted_lengths_sum (Optional[int] = None): An optional value that
        represents the total number of elements in the permuted data (output
        shape). If not provided, the operator will compute this data which may
        cause a device-host synchronization (if using GPU). Thus, it is
        recommended to supply this value to avoid such the synchronization.

Returns:
    A tuple of permuted lengths, permuted indices and permuted weights

**Example:**

    >>> permute = torch.tensor([1, 0, 2], dtype=torch.int32, device="cuda")
    >>> lengths = torch.tensor([[2, 3, 4, 5], [1, 2, 4, 8], [0, 3, 2, 3]], dtype=torch.int64, device="cuda")
    >>> values = torch.randint(low=0, high=100, size=(lengths.sum().item(),), dtype=torch.int64, device="cuda")
    >>> print(values)
    tensor([29, 12, 61, 98, 56, 94,  5, 89, 65, 48, 71, 54, 40, 33, 78, 68, 42, 21,
            60, 51, 15, 47, 48, 68, 52, 19, 38, 30, 38, 97, 97, 98, 18, 40, 42, 89,
            66], device='cuda:0')
    >>> torch.ops.fbgemm.permute_2D_sparse_data(permute, lengths, values)
    (tensor([[1, 2, 4, 8],
             [2, 3, 4, 5],
             [0, 3, 2, 3]], device='cuda:0'),
     tensor([78, 68, 42, 21, 60, 51, 15, 47, 48, 68, 52, 19, 38, 30, 38, 29, 12, 61,
             98, 56, 94,  5, 89, 65, 48, 71, 54, 40, 33, 97, 97, 98, 18, 40, 42, 89,
             66], device='cuda:0'),
     None)
    """,
)

add_docs(
    torch.ops.fbgemm.permute_1D_sparse_data,
    """
permute_1D_sparse_data(permute, lengths, values, weights=None, permuted_lengths_sum=None) -> Tuple[Tensor, Tensor, Optional[Tensor]]

Permute 1D sparse data. Note that 1D referrs to the number of dense dimensions.
The input data is actually 2D where the first dimension is dense and the second
dimension is jagged (sparse).  The data to permute over can be less or more and
withh or without repetitions.

Args:
    permute (Tensor): A 1D-tensor that describes how data is permuted along dim
        0. `permute[i]` indicates that data at position `permute[i]` is moved
        to position `i`. The length of this tensor is the total amount of data
        in dim 0 to be permuted. The values in `permute` must be >= 0 and <
        `lengths.numel()`

    lengths (Tensor): A 1D-tensor that contains jagged shapes corresponding to
        the other dense dimension. `lengths[i]` represents the jagged shape of
        data at position `i` in dim 0

    values (Tensor): A 1D-input-tensor to be permuted. The length of this
        tensor must be equal to `lengths.sum()`. This tensor can be of any data
        type.

    weights (Optional[Tensor] = None): An optional 1D-float-tensor. It must
        have the same length as `values`. It will be permuted the same way as
        values

    permuted_lengths_sum (Optional[int] = None): An optional value that
        represents the total number of elements in the permuted data (output
        shape). If not provided, the operator will compute this data which may
        cause a device-host synchronization (if using GPU). Thus, it is
        recommended to supply this value to avoid such the synchronization.

Returns:
    A tuple of permuted lengths, permuted indices and permuted weights

**Example:**
    >>> permute = torch.tensor([1, 0, 3, 0], dtype=torch.int32, device="cuda")
    >>> lengths = torch.tensor([2, 3, 4, 5], dtype=torch.int64, device="cuda")
    >>> values = torch.randint(low=0, high=100, size=(lengths.sum().item(),), dtype=torch.int64, device="cuda")
    >>> print(values)
    tensor([ 1, 76, 24, 84, 94, 25, 15, 23, 31, 46,  9, 23, 34,  3],
           device='cuda:0')
    >>> torch.ops.fbgemm.permute_1D_sparse_data(permute, lengths, values)
    (tensor([3, 2, 5, 2], device='cuda:0'),
     tensor([24, 84, 94,  1, 76, 46,  9, 23, 34,  3,  1, 76], device='cuda:0'),
     None)
    """,
)
