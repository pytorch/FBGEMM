#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from itertools import accumulate
from typing import Optional

import torch

from fbgemm_gpu.utils.loader import load_torch_module

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_cpu"
    )
    load_torch_module(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_gpu"
    )


class PermutePooledEmbeddings:
    """
    A module for permuting embedding outputs along the feature dimension

    An embedding output tensor contains the embedding outputs for all features
    in a batch. It is represented in a 2D format, where the rows are the batch
    size dimension and the columns are the feature * embedding dimension.
    Permuting along the feature dimension is essentially permuting along the
    second dimension (dim 1).

    **Example:**

        >>> import torch
        >>> import fbgemm_gpu
        >>> from fbgemm_gpu.permute_pooled_embedding_modules import PermutePooledEmbeddings
        >>>
        >>> # Suppose batch size = 3 and there are 3 features
        >>> batch_size = 3
        >>>
        >>> # Embedding dimensions for each feature
        >>> embs_dims = torch.tensor([4, 4, 8], dtype=torch.int64, device="cuda")
        >>>
        >>> # Permute list, i.e., move feature 2 to position 0, move feature 0
        >>> # to position 1, so on
        >>> permute = [2, 0, 1]
        >>>
        >>> # Instantiate the module
        >>> perm = PermutePooledEmbeddings(embs_dims, permute)
        >>>
        >>> # Generate an example input
        >>> pooled_embs = torch.arange(
        >>>     embs_dims.sum().item() * batch_size,
        >>>     dtype=torch.float32, device="cuda"
        >>> ).reshape(batch_size, -1)
        >>> print(pooled_embs)
        >>>
        tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
                 14., 15.],
                [16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
                 30., 31.],
                [32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45.,
                 46., 47.]], device='cuda:0')
        >>>
        >>> # Invoke
        >>> perm(pooled_embs)
        >>>
        tensor([[ 8.,  9., 10., 11., 12., 13., 14., 15.,  0.,  1.,  2.,  3.,  4.,  5.,
                  6.,  7.],
                [24., 25., 26., 27., 28., 29., 30., 31., 16., 17., 18., 19., 20., 21.,
                 22., 23.],
                [40., 41., 42., 43., 44., 45., 46., 47., 32., 33., 34., 35., 36., 37.,
                 38., 39.]], device='cuda:0')

    Args:
        embs_dims (List[int]): A list of embedding dimensions for all features.
            Length = the number of features

        permute (List[int]): A list that describes how each feature is
            permuted. `permute[i]` is to permute feature `permute[i]` to
            position `i`.

        device (Optional[torch.device] = None): The device to run this module
            on
    """

    def __init__(
        self,
        embs_dims: list[int],
        permute: list[int],
        device: Optional[torch.device] = None,
    ) -> None:
        self._offset_dim_list: torch.Tensor = torch.tensor(
            [0] + list(accumulate(embs_dims)), device=device, dtype=torch.int64
        )

        self._permute: torch.Tensor = torch.tensor(
            permute, device=device, dtype=torch.int64
        )

        inv_permute: list[int] = [0] * len(permute)
        for i, p in enumerate(permute):
            inv_permute[p] = i

        self._inv_permute: torch.Tensor = torch.tensor(
            inv_permute, device=device, dtype=torch.int64
        )

        inv_embs_dims = [embs_dims[i] for i in permute]

        self._inv_offset_dim_list: torch.Tensor = torch.tensor(
            [0] + list(accumulate(inv_embs_dims)), device=device, dtype=torch.int64
        )

    def __call__(self, pooled_embs: torch.Tensor) -> torch.Tensor:
        """
        Performs pooled embedding output permutation along the feature dimension

        Args:
            pooled_embs (Tensor): The embedding outputs to permute. Shape is
                `(B_local, total_global_D)`, where `B_local` = a local batch
                size and `total_global_D` is the total embedding dimension
                across all features (global)

        Returns:
            Permuted embedding outputs (Tensor). Same shape as `pooled_embs`
        """
        result = torch.ops.fbgemm.permute_pooled_embs_auto_grad(
            pooled_embs,
            self._offset_dim_list.to(device=pooled_embs.device),
            self._permute.to(device=pooled_embs.device),
            self._inv_offset_dim_list.to(device=pooled_embs.device),
            self._inv_permute.to(device=pooled_embs.device),
        )
        return result
