# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .common import add_docs

add_docs(
    torch.ops.fbgemm.permute_pooled_embs,
    """
permute_pooled_embs(pooled_embs, offset_dim_list, permute_list, inv_offset_dim_list, inv_permute_list) -> Tensor

Permute embedding outputs along the feature dimension.

The embedding output tensor `pooled_embs` contains the embedding outputs
for all features in a batch. It is represented in a 2D format, where the
rows are the batch size dimension and the columns are the feature *
embedding dimension. Permuting along the feature dimension is
essentially permuting along the second dimension (dim 1).

Args:
    pooled_embs (Tensor): The embedding outputs to permute. Shape is
        `(B_local, total_global_D)`, where `B_local` = a local batch size
        and `total_global_D` is the total embedding dimension across all
        features (global)

    offset_dim_list (Tensor): The complete cumulative sum of embedding
        dimensions of all features. Shape is `T + 1` where `T` is the
        total number of features

    permute_list (Tensor): A tensor that describes how each feature is
        permuted.  `permute_list[i]` indicates that the feature
        `permute_list[i]` is permuted to position `i`

    inv_offset_dim_list (Tensor): The complete cumulative sum of inverse
        embedding dimensions, which are the permuted embedding dimensions.
        `inv_offset_dim_list[i]` represents the starting embedding position of
        feature `permute_list[i]`

    inv_permute_list (Tensor): The inverse permute list, which contains the
        permuted positions of each feature. `inv_permute_list[i]` represents
        the permuted position of feature `i`

Returns:
    Permuted embedding outputs (Tensor). Same shape as `pooled_embs`

**Example:**

    >>> import torch
    >>> from itertools import accumulate
    >>>
    >>> # Suppose batch size = 3 and there are 3 features
    >>> batch_size = 3
    >>>
    >>> # Embedding dimensions for each feature
    >>> embs_dims = torch.tensor([4, 4, 8], dtype=torch.int64, device="cuda")
    >>>
    >>> # Permute list, i.e., move feature 2 to position 0, move feature 0
    >>> # to position 1, so on
    >>> permute = torch.tensor([2, 0, 1], dtype=torch.int64, device="cuda")
    >>>
    >>> # Compute embedding dim offsets
    >>> offset_dim_list = torch.tensor([0] + list(accumulate(embs_dims)), dtype=torch.int64, device="cuda")
    >>> print(offset_dim_list)
    >>>
    tensor([ 0,  4,  8, 16], device='cuda:0')
    >>>
    >>> # Compute inverse embedding dims
    >>> inv_embs_dims = [embs_dims[p] for p in permute]
    >>> # Compute complete cumulative sum of inverse embedding dims
    >>> inv_offset_dim_list = torch.tensor([0] + list(accumulate(inv_embs_dims)), dtype=torch.int64, device="cuda")
    >>> print(inv_offset_dim_list)
    >>>
    tensor([ 0,  8, 12, 16], device='cuda:0')
    >>>
    >>> # Compute inverse permutes
    >>> inv_permute = [0] * len(permute)
    >>> for i, p in enumerate(permute):
    >>>     inv_permute[p] = i
    >>> inv_permute_list = torch.tensor([inv_permute], dtype=torch.int64, device="cuda")
    >>> print(inv_permute_list)
    >>>
    tensor([[1, 2, 0]], device='cuda:0')
    >>>
    >>> # Generate an example input
    >>> pooled_embs = torch.arange(embs_dims.sum().item() * batch_size, dtype=torch.float32, device="cuda").reshape(batch_size, -1)
    >>> print(pooled_embs)
    >>>
    tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
             14., 15.],
            [16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
             30., 31.],
            [32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45.,
             46., 47.]], device='cuda:0')
    >>>
    >>> torch.ops.fbgemm.permute_pooled_embs_auto_grad(pooled_embs, offset_dim_list, permute, inv_offset_dim_list, inv_permute_list)
    >>>
    tensor([[ 8.,  9., 10., 11., 12., 13., 14., 15.,  0.,  1.,  2.,  3.,  4.,  5.,
              6.,  7.],
            [24., 25., 26., 27., 28., 29., 30., 31., 16., 17., 18., 19., 20., 21.,
             22., 23.],
            [40., 41., 42., 43., 44., 45., 46., 47., 32., 33., 34., 35., 36., 37.,
             38., 39.]], device='cuda:0')
    """,
)
