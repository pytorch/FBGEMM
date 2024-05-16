#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .split_embeddings_cache_ops import get_unique_indices

lib = torch.library.Library("fbgemm", "FRAGMENT")
lib.define(
    """
    get_unique_indices(
        Tensor linear_indices,
        int max_indices,
        bool compute_count=False,
        bool compute_inverse_indices=False
    ) -> (Tensor, Tensor, Tensor?, Tensor?)
    """
)

lib.impl("get_unique_indices", get_unique_indices, "CUDA")
