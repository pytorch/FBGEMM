# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

from typing import Any, List, Tuple

import fbgemm_gpu
import torch
from fbgemm_gpu.permute_pooled_embedding_modules import PermutePooledEmbeddings
from hypothesis import HealthCheck
from torch import Tensor

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, on_arm_platform
else:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_split_gpu"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_split_cpu"
    )
    from fbgemm_gpu.test.test_utils import gpu_unavailable, on_arm_platform

typed_gpu_unavailable: Tuple[bool, str] = gpu_unavailable
typed_on_arm_platform: Tuple[bool, str] = on_arm_platform

suppressed_list: List[HealthCheck] = (
    [HealthCheck.not_a_test_method]
    if getattr(HealthCheck, "not_a_test_method", False)
    else []
) + (
    [HealthCheck.differing_executors]
    if getattr(HealthCheck, "differing_executors", False)
    else []
)

INTERN_MODULE = "fbgemm_gpu.permute_pooled_embedding_modules"
FIXED_EXTERN_API = {
    "PermutePooledEmbeddings": {
        "__init__": ["self", "embs_dims", "permute", "device"],
        "__call__": ["self", "pooled_embs"],
    },
}

FWD_COMPAT_MSG = (
    "WARNING: If this test is failing, you are probably trying "
    "to make changes to a module that has been marked external to PyPer torch packages. "
    "This can break forward compatibility of torch packages on training_platform "
    "(see https://fb.workplace.com/groups/pyper/permalink/808155810065803/). "
    "You need to split up your changes as follows:\n"
    "\t1. Edit your diff so it only contains the changes as optional, and not any usage of the"
    " new optional changes.\n"
    "\t2. Edit FIXED_EXTERN_API in this test so your diff passes the test.\n"
    "\t3. Land your diff and wait for the diff to be picked up by the production version of"
    " fbpkg training_platform.\n"
    "\t4. Once step 3. is complete, you can push the rest of your changes that use the new"
    " changes."
)


class PermutePooledEmbeddingsFwdOnly(PermutePooledEmbeddings):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, pooled_embs: torch.Tensor) -> torch.Tensor:
        result = torch.ops.fbgemm.permute_pooled_embs(
            pooled_embs,
            self._offset_dim_list.to(device=pooled_embs.device),
            self._permute.to(device=pooled_embs.device),
            self._inv_offset_dim_list.to(device=pooled_embs.device),
            self._inv_permute.to(device=pooled_embs.device),
        )
        return result


class Net(torch.nn.Module):
    def __init__(self, fwd_only: bool = False) -> None:
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 10, bias=False)
        op_cls = PermutePooledEmbeddingsFwdOnly if fwd_only else PermutePooledEmbeddings
        self.permute_pooled_embeddings: PermutePooledEmbeddings = op_cls(
            [2, 3, 1, 4],
            [3, 0, 2, 1],
        )
        self.fc2 = torch.nn.Linear(10, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.permute_pooled_embeddings(x)
        x = self.fc2(x)
        return x
