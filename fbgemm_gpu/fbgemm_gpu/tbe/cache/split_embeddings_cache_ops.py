# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional, Tuple, Union

import torch

lib = torch.library.Library("fbgemm", "FRAGMENT")
lib.define(
    """
    get_unique_indices_v2(
        Tensor linear_indices,
        int max_indices,
        bool compute_count=False,
        bool compute_inverse_indices=False
    ) -> (Tensor, Tensor, Tensor?, Tensor?)
    """
)


@torch.library.impl(lib, "get_unique_indices_v2", "CUDA")
def get_unique_indices_v2(
    linear_indices: torch.Tensor,
    max_indices: int,
    compute_count: bool = False,
    compute_inverse_indices: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
    Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ],
]:
    """
    A wrapper for get_unique_indices for overloading the return type
    based on inputs
    """
    ret = torch.ops.fbgemm.get_unique_indices_with_inverse(
        linear_indices,
        max_indices,
        compute_count,
        compute_inverse_indices,
    )
    if compute_count and compute_inverse_indices:
        # Return all tensors
        return ret
    if compute_count:
        # Return (unique_indices, length, count)
        return ret[:-1]
    if compute_inverse_indices:
        # Return (unique_indices, length, inverse_indices)
        # pyre-fixme[7]: The arity arity of this return is wrong (3 vs 4)
        return ret[0], ret[1], ret[3]
    # Return (unique_indices, length)
    return ret[:-2]
