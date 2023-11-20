#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from itertools import accumulate
from typing import List, Optional

import torch

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_cpu"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_gpu"
    )


class PermutePooledEmbeddings:
    def __init__(
        self,
        embs_dims: List[int],
        permute: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        logging.info("Using Permute Pooled Embeddings")
        self._offset_dim_list: torch.Tensor = torch.tensor(
            [0] + list(accumulate(embs_dims)), device=device, dtype=torch.int64
        )

        self._permute: torch.Tensor = torch.tensor(
            permute, device=device, dtype=torch.int64
        )

        inv_permute: List[int] = [0] * len(permute)
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
        result = torch.ops.fbgemm.permute_pooled_embs_auto_grad(
            pooled_embs,
            self._offset_dim_list.to(device=pooled_embs.device),
            self._permute.to(device=pooled_embs.device),
            self._inv_offset_dim_list.to(device=pooled_embs.device),
            self._inv_permute.to(device=pooled_embs.device),
        )
        return result
