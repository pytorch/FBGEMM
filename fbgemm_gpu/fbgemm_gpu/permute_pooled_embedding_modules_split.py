#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from itertools import accumulate
from typing import Optional

import torch
from torch import nn

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_split_gpu"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_split_cpu"
    )


@torch.fx.wrap
def _fx_wrap_tensor_to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.to(device=device)


class PermutePooledEmbeddingsSplit(nn.Module):
    def __init__(
        self,
        embs_dims: list[int],
        permute: list[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super(PermutePooledEmbeddingsSplit, self).__init__()
        logging.info("Using Permute Pooled Embeddings")

        self.register_buffer(
            "_offset_dim_list",
            torch.tensor(
                [0] + list(accumulate(embs_dims)), device=device, dtype=torch.int64
            ),
        )
        self.register_buffer(
            "_permute", torch.tensor(permute, device=device, dtype=torch.int64)
        )

        inv_permute: list[int] = [0] * len(permute)
        for i, p in enumerate(permute):
            inv_permute[p] = i

        self.register_buffer(
            "_inv_permute", torch.tensor(inv_permute, device=device, dtype=torch.int64)
        )

        #  `Union[BoundMethod[typing.Callable(torch.Tensor.tolist)[[Named(self,
        #  torch.Tensor)], List[typing.Any]], torch.Tensor], nn.Module, torch.Tensor]`
        #  is not a function.

        inv_embs_dims = [embs_dims[i] for i in permute]

        self.register_buffer(
            "_inv_offset_dim_list",
            torch.tensor(
                [0] + list(accumulate(inv_embs_dims)), device=device, dtype=torch.int64
            ),
        )

    def forward(self, pooled_embs: torch.Tensor) -> torch.Tensor:
        result = torch.ops.fbgemm.permute_pooled_embs_auto_grad_split(
            pooled_embs,
            _fx_wrap_tensor_to_device(self._offset_dim_list, device=pooled_embs.device),
            _fx_wrap_tensor_to_device(self._permute, device=pooled_embs.device),
            _fx_wrap_tensor_to_device(
                self._inv_offset_dim_list, device=pooled_embs.device
            ),
            _fx_wrap_tensor_to_device(self._inv_permute, device=pooled_embs.device),
        )
        return result
