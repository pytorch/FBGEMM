# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union

import torch

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


@torch.library.impl(lib, "get_unique_indices", "CUDA")
def get_unique_indices(
    linear_indices: torch.Tensor,
    max_indices: int,
    compute_count: bool = False,
    compute_inverse_indices: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
]:
    """
    A wrapper for get_unique_indices for overloading the return type
    based on inputs
    """
    ret = torch.ops.fbgemm.get_unique_indices_internal(
        linear_indices,
        max_indices,
        compute_count,
        compute_inverse_indices,
    )
    if not compute_inverse_indices:
        # Return only 3 tensors
        return ret[:-1]
    # Return all tensors
    return ret
