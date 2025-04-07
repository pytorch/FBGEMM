# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

import numpy as np
import torch

from fbgemm_gpu.split_embedding_configs import SparseType

logging.basicConfig(level=logging.DEBUG)


def fill_random_scale_bias(
    emb: torch.nn.Module,
    T: int,
    weights_precision: SparseType,
) -> None:
    for t in range(T):
        # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        (weights, scale_shift) = emb.split_embedding_weights()[t]
        if scale_shift is not None:
            (E, R) = scale_shift.shape
            assert R == 4
            scales = None
            shifts = None
            if weights_precision == SparseType.INT8:
                scales = np.random.uniform(0.001, 0.01, size=(E,)).astype(np.float16)
                shifts = np.random.normal(-2, 2, size=(E,)).astype(np.float16)
            elif weights_precision == SparseType.INT4:
                scales = np.random.uniform(0.01, 0.1, size=(E,)).astype(np.float16)
                shifts = np.random.normal(-2, 2, size=(E,)).astype(np.float16)
            elif weights_precision == SparseType.INT2:
                scales = np.random.uniform(0.1, 1, size=(E,)).astype(np.float16)
                shifts = np.random.normal(-2, 2, size=(E,)).astype(np.float16)
            scale_shift.copy_(
                torch.tensor(
                    np.stack([scales, shifts], axis=1)
                    .astype(np.float16)
                    .view(np.uint8),
                    device=scale_shift.device,
                )
            )
