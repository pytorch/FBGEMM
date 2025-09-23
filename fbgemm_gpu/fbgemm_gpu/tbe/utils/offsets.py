# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, Optional

import numpy as np
import torch

from .common import to_device


# Merged indices with shape (T, B, L) -> (flattened indices with shape
# (T * B * L), offsets with shape (T * B + 1))
def get_table_batched_offsets_from_dense(
    merged_indices: torch.Tensor,
    L: Optional[int] = None,
    total_B: Optional[int] = None,
    use_cpu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if L is None and total_B is None:
        (T, B, L) = merged_indices.size()
        total_B = T * B
    # pyre-fixme[6]: For 1st argument expected `Union[Sequence[SupportsIndex],
    #  SupportsIndex]` but got `Optional[int]`.
    lengths = np.ones(total_B) * L
    return (
        to_device(merged_indices.contiguous().view(-1), use_cpu),
        to_device(
            torch.tensor(([0] + np.cumsum(lengths).tolist())).long(),
            use_cpu,
        ),
    )


def get_offsets_from_dense(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    (B, L) = indices.size()
    return (
        indices.contiguous().view(-1),
        torch.tensor(
            np.cumsum(np.asarray([0] + [L for _ in range(B)])[:-1]).astype(np.int64)
        ),
    )


def b_indices(
    b: Callable[..., torch.Tensor],
    x: torch.Tensor,
    per_sample_weights: Optional[torch.Tensor] = None,
    use_cpu: bool = False,
    do_pooling: bool = True,
) -> torch.Tensor:
    (indices, offsets) = get_offsets_from_dense(x)
    if do_pooling:
        return b(
            to_device(indices, use_cpu),
            to_device(offsets, use_cpu),
            per_sample_weights=per_sample_weights,
        )
    else:
        return b(to_device(indices, use_cpu))
