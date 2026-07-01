#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Optimizer-specific enums and frozen dataclasses for TBE.

These types configure the optimizer behavior of Split Table Batched Embeddings.
They are used exclusively at init-time — the __init__() method decomposes them
into primitive scalar self.* attributes before JIT scripting.

These types MUST stay ``@dataclass``: they are used as field types in downstream
OmegaConf / Hydra structured configs (e.g. torchrec optimizer configs), and
OmegaConf only accepts dataclass / attrs classes as nested config nodes — a
``NamedTuple`` field type raises ``omegaconf.errors.ValidationError: Unexpected
type annotation`` and breaks those config builds.

torch.package safety: DO NOT ADD
``from __future__ import annotations`` TO THIS MODULE, and do not give the
``@dataclass`` types below string annotations. These dataclasses are eagerly
re-exported from ``fbgemm_gpu/tbe/config/__init__.py`` (and from
``split_table_batched_embeddings_ops_training``), so their source is interned into
every publisher's torch.package archive and re-executed under the
``<torch_package_N>`` sandbox when an already-published model is re-loaded /
re-published. With *string* annotations (which ``from __future__ import
annotations`` forces), ``@dataclass`` decoration calls ``dataclasses._is_type`` ->
``sys.modules.get(cls.__module__).__dict__``; the mangled sandbox module is not
registered in ``sys.modules``, so that is ``None.__dict__`` -> ``AttributeError``
at decoration time (the model republish crashes). With *eager* (non-string)
annotations — the current state — that path is never taken, so these dataclasses
are torch.package-safe. Keeping annotations eager is load-bearing; note this
conflicts with the general repo guidance to add ``from __future__ import
annotations`` for lazy imports.
"""

import enum
from dataclasses import dataclass, field


class DoesNotHavePrefix(Exception):
    pass


class WeightDecayMode(enum.IntEnum):
    NONE = 0
    L2 = 1
    DECOUPLE = 2
    COUNTER = 3
    COWCLIP = 4
    DECOUPLE_GLOBAL = 5


class CounterWeightDecayMode(enum.IntEnum):
    NONE = 0
    L2 = 1
    DECOUPLE = 2
    ADAGRADW = 3


class StepMode(enum.IntEnum):
    NONE = 0
    USE_COUNTER = 1
    USE_ITER = 2


class LearningRateMode(enum.IntEnum):
    EQUAL = -1
    TAIL_ID_LR_INCREASE = 0
    TAIL_ID_LR_DECREASE = 1
    COUNTER_SGD = 2


class GradSumDecay(enum.IntEnum):
    NO_DECAY = -1
    CTR_DECAY = 0


@dataclass(frozen=True)
class TailIdThreshold:
    val: float = 0
    is_ratio: bool = False


@dataclass(frozen=True)
class CounterBasedRegularizationDefinition:
    counter_weight_decay_mode: CounterWeightDecayMode = CounterWeightDecayMode.NONE
    counter_halflife: int = -1
    adjustment_iter: int = -1
    adjustment_ub: float = 1.0
    learning_rate_mode: LearningRateMode = LearningRateMode.EQUAL
    grad_sum_decay: GradSumDecay = GradSumDecay.NO_DECAY
    tail_id_threshold: TailIdThreshold = field(default_factory=TailIdThreshold)
    max_counter_update_freq: int = 1000


@dataclass(frozen=True)
class CowClipDefinition:
    counter_weight_decay_mode: CounterWeightDecayMode = CounterWeightDecayMode.NONE
    counter_halflife: int = -1
    weight_norm_coefficient: float = 0.0
    lower_bound: float = 0.0


@dataclass(frozen=True)
class GlobalWeightDecayDefinition:
    start_iter: int = 0
    lower_bound: float = 0.0


@dataclass(frozen=True)
class UserEnabledConfigDefinition:
    """
    This class is used to configure whether certain modes are to be enabled
    """

    # This is used in Adam to perform rowwise bias correction using `row_counter`
    # More details can be found in D64848802.
    use_rowwise_bias_correction: bool = False
    use_writeback_bwd_prehook: bool = False
    writeback_first_feature_only: bool = False
    precompute_writeback: bool = False


@dataclass(frozen=True)
class EnsembleModeDefinition:
    step_ema: float = 10000
    step_swap: float = 10000
    step_start: float = 0
    step_ema_coef: float = 0.6
    step_mode: StepMode = StepMode.USE_ITER


@dataclass(frozen=True)
class EmainplaceModeDefinition:
    step_ema: float = 10
    step_start: float = 0
    step_ema_coef: float = 0.6
