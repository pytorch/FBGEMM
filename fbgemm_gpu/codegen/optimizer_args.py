#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple

import torch
from torch import nn


class SplitEmbeddingOptimizerParams(NamedTuple):
    weights_dev: nn.Parameter
    # TODO: Enable weights_uvm and weights_lxu_cache support
    # weights_uvm: nn.Parameter
    # weights_lxu_cache: nn.Parameter


class SplitEmbeddingArgs(NamedTuple):
    weights_placements: torch.Tensor
    weights_offsets: torch.Tensor
    max_D: int
