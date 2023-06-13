#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple, Optional

import torch


class VBEMetadata(NamedTuple):
    B_offsets: Optional[torch.Tensor]
    output_offsets_feature_rank: Optional[torch.Tensor]
    B_offsets_rank_per_feature: Optional[torch.Tensor]
    max_B_feature_rank: int = -1
    max_B: int = -1
    output_size: int = -1


class CommonArgs(NamedTuple):
    placeholder_autograd_tensor: torch.Tensor
    dev_weights: torch.Tensor
    host_weights: torch.Tensor
    uvm_weights: torch.Tensor
    lxu_cache_weights: torch.Tensor
    weights_placements: torch.Tensor
    weights_offsets: torch.Tensor
    D_offsets: torch.Tensor
    total_D: int
    max_D: int
    hash_size_cumsum: torch.Tensor
    total_hash_size_bits: int
    indices: torch.Tensor
    offsets: torch.Tensor
    pooling_mode: int
    indice_weights: Optional[torch.Tensor]
    feature_requires_grad: Optional[torch.Tensor]
    lxu_cache_locations: torch.Tensor
    output_dtype: int
    vbe_metadata: VBEMetadata
    is_experimental: bool


class OptimizerArgs(NamedTuple):
    stochastic_rounding: bool
    gradient_clipping: bool
    max_gradient: float
    learning_rate: float
    eps: float
    beta1: float
    beta2: float
    weight_decay: float
    weight_decay_mode: int
    eta: float
    momentum: float
    counter_halflife: int
    adjustment_iter: int
    adjustment_ub: float
    learning_rate_mode: int
    grad_sum_decay: int
    tail_id_threshold: float
    is_tail_id_thresh_ratio: int


class Momentum(NamedTuple):
    dev: torch.Tensor
    host: torch.Tensor
    uvm: torch.Tensor
    offsets: torch.Tensor
    placements: torch.Tensor
