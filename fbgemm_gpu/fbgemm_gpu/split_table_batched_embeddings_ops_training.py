#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import contextlib
import enum
import functools
import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from itertools import accumulate
from math import log2
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch  # usort:skip
from torch import nn, Tensor  # usort:skip

# @manual=//deeplearning/fbgemm/fbgemm_gpu/codegen:split_embedding_codegen_lookup_invokers
import fbgemm_gpu.split_embedding_codegen_lookup_invokers as invokers

from fbgemm_gpu.config import FeatureGate, FeatureGateName
from fbgemm_gpu.runtime_monitor import (
    AsyncSeriesTimer,
    TBEStatsReporter,
    TBEStatsReporterConfig,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheState,
    ComputeDevice,
    construct_cache_state,
    EmbeddingLocation,
    get_bounds_check_version_for_platform,
    MAX_PREFETCH_DEPTH,
    MultiPassPrefetchConfig,
    PoolingMode,
    RecordCacheMetrics,
    SplitState,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import (
    generate_vbe_metadata,
    is_torchdynamo_compiling,
)
from fbgemm_gpu.tbe_input_multiplexer import (
    TBEInfo,
    TBEInputInfo,
    TBEInputMultiplexer,
    TBEInputMultiplexerConfig,
)

from fbgemm_gpu.utils.loader import load_torch_module, load_torch_module_bc

try:
    load_torch_module(
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_training_gpu",
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cuda_training",
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_hip_training",
    )
    load_torch_module_bc(
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_training_cpu",
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cpu_training",
    )
except Exception:
    pass


DEFAULT_ASSOC = 32 if torch.version.hip is None else 64
INT8_EMB_ROW_DIM_OFFSET = 8


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


# Keep in sync with fbgemm_gpu/include/fbgemm_gpu/split_embeddings_cache_cuda.cuh
class UVMCacheStatsIndex(enum.IntEnum):
    num_calls = 0
    num_requested_indices = 1
    num_unique_indices = 2
    num_unique_misses = 3
    num_conflict_unique_misses = 4
    num_conflict_misses = 5


@dataclass
class RESParams:
    res_server_port: int = 0  # the port of the res server
    res_store_shards: int = 1  # the number of shards to store the raw embeddings
    table_names: List[str] = field(default_factory=list)  # table names the TBE holds
    table_offsets: List[int] = field(
        default_factory=list
    )  # table offsets for the global rows the TBE holds
    table_sizes: List[int] = field(
        default_factory=list
    )  # table sizes for the global rows the TBE holds


def construct_split_state(
    embedding_specs: List[Tuple[int, int, EmbeddingLocation, ComputeDevice]],
    rowwise: bool,
    cacheable: bool,
    precision: SparseType = SparseType.FP32,
    int8_emb_row_dim_offset: int = INT8_EMB_ROW_DIM_OFFSET,
    placement: Optional[EmbeddingLocation] = None,
) -> SplitState:
    placements: List[EmbeddingLocation] = []
    offsets: List[int] = []
    dev_size: int = 0
    host_size: int = 0
    uvm_size: int = 0
    for num_embeddings, embedding_dim, location, _ in embedding_specs:
        assert (
            embedding_dim % 4 == 0
        ), f"embedding_dim must be a multiple of 4, but got {embedding_dim}"
        if precision == SparseType.INT8:
            embedding_dim += int8_emb_row_dim_offset
        state_size = num_embeddings * embedding_dim if not rowwise else num_embeddings
        location = placement if placement is not None else location
        if location == EmbeddingLocation.HOST:
            placements.append(EmbeddingLocation.HOST)
            offsets.append(host_size)
            host_size += state_size
        # If table is on device, then opimtizer is on device.
        # If table is managed, then if optimizer state is rowwise, optimizer is on device, otherwise optimizer is managed.
        elif location == EmbeddingLocation.DEVICE or rowwise:
            placements.append(EmbeddingLocation.DEVICE)
            offsets.append(dev_size)
            dev_size += state_size
        else:
            if cacheable and location == EmbeddingLocation.MANAGED_CACHING:
                placements.append(EmbeddingLocation.MANAGED_CACHING)
            else:
                placements.append(EmbeddingLocation.MANAGED)
            offsets.append(uvm_size)
            uvm_size += state_size
    assert len(placements) == len(offsets)
    return SplitState(
        dev_size=dev_size,
        host_size=host_size,
        uvm_size=uvm_size,
        placements=placements,
        offsets=offsets,
    )


def apply_split_helper(
    persistent_state_fn: Callable[[str, Tensor], None],
    set_attr_fn: Callable[
        [str, Union[Tensor, List[int], List[EmbeddingLocation]]], None
    ],
    current_device: torch.device,
    use_cpu: bool,
    feature_table_map: List[int],
    split: SplitState,
    prefix: str,
    dtype: Type[torch.dtype],
    enforce_hbm: bool = False,
    make_dev_param: bool = False,
    dev_reshape: Optional[Tuple[int, ...]] = None,
    uvm_tensors_log: Optional[List[str]] = None,
    uvm_host_mapped: bool = False,
) -> None:
    set_attr_fn(f"{prefix}_physical_placements", split.placements)
    set_attr_fn(f"{prefix}_physical_offsets", split.offsets)

    offsets = [split.offsets[t] for t in feature_table_map]
    placements = [split.placements[t] for t in feature_table_map]
    persistent_state_fn(
        f"{prefix}_offsets",
        torch.tensor(offsets, device=current_device, dtype=torch.int64),
    )
    persistent_state_fn(
        f"{prefix}_placements",
        torch.tensor(placements, device=current_device, dtype=torch.int32),
    )
    if split.dev_size > 0:
        dev_buffer = torch.zeros(
            split.dev_size,
            device=current_device,
            # pyre-fixme[6]
            dtype=dtype,
        )
        dev_buffer = (
            dev_buffer.view(*dev_reshape) if dev_reshape is not None else dev_buffer
        )
    else:
        # pyre-fixme[6]
        dev_buffer = torch.empty(0, device=current_device, dtype=dtype)
    if make_dev_param:
        set_attr_fn(f"{prefix}_dev", nn.Parameter(dev_buffer))
    else:
        persistent_state_fn(f"{prefix}_dev", dev_buffer)
    if split.host_size > 0:
        if dtype == torch.uint8:
            persistent_state_fn(
                f"{prefix}_host",
                torch.zeros(
                    split.host_size,
                    device=current_device,
                    # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for
                    #  3rd param but got `Type[Type[torch._dtype]]`.
                    dtype=dtype,
                ),
            )
        else:
            set_attr_fn(
                f"{prefix}_host",
                nn.Parameter(
                    torch.zeros(
                        split.host_size,
                        device=current_device,
                        # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]`
                        #  for 3rd param but got `Type[Type[torch._dtype]]`.
                        dtype=dtype,
                    )
                ),
            )
        if uvm_tensors_log is not None:
            uvm_tensors_log.append(f"{prefix}_host")
    else:
        persistent_state_fn(
            f"{prefix}_host",
            # pyre-fixme[6]: For 3rd param expected `dtype` but got `Type[dtype]`.
            torch.empty(0, device=current_device, dtype=dtype),
        )
    if split.uvm_size > 0:
        assert not use_cpu
        if enforce_hbm:
            logging.info("Enforce hbm for the cache location")
            persistent_state_fn(
                f"{prefix}_uvm",
                torch.zeros(
                    split.uvm_size,
                    device=current_device,
                    # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for
                    #  3rd param but got `Type[Type[torch._dtype]]`.
                    dtype=dtype,
                ),
            )
        else:
            persistent_state_fn(
                f"{prefix}_uvm",
                torch.zeros(
                    split.uvm_size,
                    out=torch.ops.fbgemm.new_unified_tensor(
                        # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]`
                        #  for 3rd param but got `Type[Type[torch._dtype]]`.
                        torch.zeros(1, device=current_device, dtype=dtype),
                        [split.uvm_size],
                        is_host_mapped=uvm_host_mapped,
                    ),
                ),
            )
            if uvm_tensors_log is not None:
                uvm_tensors_log.append(f"{prefix}_uvm")
    else:
        persistent_state_fn(
            f"{prefix}_uvm",
            # pyre-fixme[6]: For 3rd param expected `dtype` but got `Type[dtype]`.
            torch.empty(0, device=current_device, dtype=dtype),
        )


def get_available_compute_device() -> ComputeDevice:
    if torch.cuda.is_available():
        return ComputeDevice.CUDA
    elif torch.mtia.is_available():
        return ComputeDevice.MTIA
    else:
        return ComputeDevice.CPU


# pyre-fixme[13]: Attribute `uvm_cache_stats` is never initialized.
# pyre-fixme[13]: Attribute `local_uvm_cache_stats` is never initialized.
class SplitTableBatchedEmbeddingBagsCodegen(nn.Module):
    """
    Table Batched Embedding (TBE) operator.  Looks up one or more embedding
    tables. The module is application for training. The backward operator is
    fused with optimizer. Thus, the embedding tables are updated during
    backward.

    Args:
        embedding_specs (List[Tuple[int, int, EmbeddingLocation, ComputeDevice]]):
            A list of embedding specifications. Each spec describes a
            specification of a physical embedding table. Each one is a tuple of
            number of embedding rows, embedding dimension (must be a multiple of
            4), table placement (`EmbeddingLocation`), and compute device
            (`ComputeDevice`).

            Available `EmbeddingLocation` options are

            (1) `DEVICE` = placing an embedding table in the GPU global memory
                (HBM)

            (2) `MANAGED` = placing an embedding in the unified virtual memory
                (accessible from both GPU and CPU)

            (3) `MANAGED_CACHING` = placing an embedding table in the unified
                virtual memory and using the GPU global memory (HBM) as a cache

            (4) `HOST` = placing an embedding table in the CPU memory (DRAM)

            (5) `MTIA` = placing an embedding table in the MTIA memory

            Available `ComputeDevice` options are

            (1) `CPU` = performing table lookup on CPU

            (2) `CUDA` = performing table lookup on GPU

            (3) `MTIA` = performing table lookup on MTIA

        feature_table_map (Optional[List[int]] = None): An optional list that
            specifies feature-table mapping. feature_table_map[i] indicates the
            physical embedding table that feature i maps to.

        cache_algorithm (CacheAlgorithm = CacheAlgorithm.LRU): The cache
            algorithm (used when `EmbeddingLocation` is set to
            `MANAGED_CACHING`).  Options are

            (1) `LRU` = least recently used

            (2) `LFU` = least frequently used

        cache_load_factor (float = 0.2): A factor used for determining the
            cache capacity when `EmbeddingLocation.MANAGED_CACHING` is used.
            The cache capacity is `cache_load_factor` * the total number of
            rows in all embedding tables

        cache_sets (int = 0): The number of cache sets (used when
            `EmbeddingLocation` is set to `MANAGED_CACHING`)

        cache_reserved_memory (float = 0.0): The amount of memory reserved in
            HBM for non-cache purpose (used when `EmbeddingLocation` is set to
            `MANAGED_CACHING`).

        cache_precision (SparseType = SparseType.FP32): The data type of the
            cache (used when `EmbeddingLocation` is set to `MANAGED_CACHING`).
            Options are `SparseType.FP32` and `SparseType.FP16`

        weights_precision (SparseType = SparseType.FP32): The data type of
            embedding tables (also known as weights). Options are
            `SparseType.FP32` and `SparseType.FP16`

        output_dtype (SparseType = SparseType.FP32): The data type of an output
            tensor. Options are `SparseType.FP32` and `SparseType.FP16`

        enforce_hbm (bool = False): If True, place all weights/momentums in HBM
            when using `EmbeddingLocation.MANAGED_CACHING`

        optimizer (OptimType = OptimType.EXACT_SGD): An optimizer to use for
            embedding table update in the backward pass.  Available `OptimType`
            options are

            (1) `ADAM` = Adam

            (2) `EXACT_ADAGRAD` = Adagrad

            (3) `EXACT_ROWWISE_ADAGRAD` = Rowwise-Aadagrad

            (4) `EXACT_SGD` = SGD

            (5) `LAMB` = Lamb

            (6) `LARS_SGD` = LARS-SGD

            (7) `PARTIAL_ROWWISE_ADAM` = Partial rowwise-Adam

            (8) `PARTIAL_ROWWISE_LAMB` = Partial rowwise-Lamb

            (9) `ENSEMBLE_ROWWISE_ADAGRAD` = Ensemble rowwise-Adagrad

            (10) `EMAINPLACE_ROWWISE_ADAGRAD` = Ema inplace rowwise-Adagrad

            (11) `NONE` = Not applying an optimizer update in the backward pass
                and outputting a sparse weight gradient

        record_cache_metrics (Optional[RecordCacheMetrics] = None): Record
            a number of hits, a number of requests, etc if
            `RecordCacheMetrics.record_cache_miss_counter` is True and record
            the similar metrics table-wise if
            `RecordCacheMetrics.record_tablewise_cache_miss is True`

        gather_uvm_cache_stats (Optional[bool] = False): If True, collect the
            cache statistics when `EmbeddingLocation` is set to
            `MANAGED_CACHING`

        stochastic_rounding (bool = True): If True, apply stochastic rounding
            for weight type that is not `SparseType.FP32`

        gradient_clipping (bool = False): If True, apply gradient clipping

        max_gradient (float = 1.0): The value for gradient clipping

        max_norm (float = 0.0): The max norm value

        learning_rate (float = 0.01): The learning rate

        eps (float = 1.0e-8): The epsilon value used by Adagrad, LAMB, and
            Adam. Note that default is different from torch.nn.optim.Adagrad
            default of 1e-10

        momentum (float = 0.9): Momentum used by LARS-SGD

        weight_decay (float = 0.0): Weight decay used by LARS-SGD, LAMB, ADAM,
            and rowwise-Adagrad.

            (1) EXACT_ADAGRAD, SGD, EXACT_SGD do not support weight decay

            (2) LAMB, ADAM, PARTIAL_ROWWISE_ADAM, PARTIAL_ROWWISE_LAMB, LARS_SGD
                support decoupled weight decay

            (3) EXACT_ROWWISE_ADAGRAD support both L2 and decoupled weight decay
                (via weight_decay_mode)

        weight_decay_mode (WeightDecayMode = WeightDecayMode.NONE): Weight decay
            mode. Options are `WeightDecayMode.NONE`, `WeightDecayMode.L2`,
            and `WeightDecayMode.DECOUPLE`

        eta (float = 0.001): The eta value used by LARS-SGD

        beta1 (float = 0.9): The beta1 value used by LAMB and ADAM

        beta2 (float = 0.999): The beta2 value used by LAMB and ADAM

        ensemble_mode (Optional[EnsembleModeDefinition] = None):
            Used by Ensemble Rowwise Adagrad

        emainplace_mode (Optional[EmainplaceModeDefinition] = None):
            Used by EMA in-place Rowwise Adagrad

        counter_based_regularization (Optional[CounterBasedRegularizationDefinition] = None):
            Used by Rowwise Adagrad

        cowclip_regularization (Optional[CowClipDefinition] = None): Used by
            Rowwise Adagrad

        pooling_mode (PoolingMode = PoolingMode.SUM): Pooling mode. Available
            `PoolingMode` options are

            (1) `SUM` = Sum pooling

            (2) `MEAN` = Mean pooling

            (3) `NONE` = No pooling (sequence embedding)

        device (Optional[Union[str, int, torch.device]] = None): The current
            device to place tensors on

        bounds_check_mode (BoundsCheckMode = BoundsCheckMode.WARNING): Input
            checking mode. Available `BoundsCheckMode` options are

            (1) `NONE` = skip bounds check

            (2) `FATAL` = throw an error when encountering an invalid
                index/offset

            (3) `WARNING` = print a warning message when encountering an
                invalid index/offset and fix it (setting an invalid index to
                zero and adjusting an invalid offset to be within the bound)

            (4) `IGNORE` = silently fix an invalid index/offset (setting an
                invalid index to zero and adjusting an invalid offset to be
                within the bound)

        uvm_non_rowwise_momentum (bool = False): If True, place non-rowwise
            momentum on the unified virtual memory

        use_experimental_tbe (bool = False): If True, use an optimized TBE
            implementation (TBE v2). Note that this is supported only on NVIDIA
            GPUs.

        prefetch_pipeline (bool = False): If True, enable cache prefetch
            pipeline when using `EmbeddingLocation.MANAGED_CACHING`. Currently
            only supports the LRU cache policy. If a separate stream is used
            for prefetch, the optional `forward_stream` arg of prefetch
            function must be set.

        stats_reporter_config (Optional[TBEStatsReporterConfig] = None):
            A config for TBE stats reporter

        table_names (Optional[List[str]] = None): A list of embedding table
            names in this TBE

        optimizer_state_dtypes (Optional[Dict[str, SparseType]] = None): A
            optimizer state data types dict. Keys are the optimizer state names
            and values are their corresponding types

        multipass_prefetch_config (Optional[MultiPassPrefetchConfig] = None):
            A config for multipass cache prefetching (when
            `EmbeddingLocation.MANAGED_CACHING` is used)

        global_weight_decay (Optional[GlobalWeightDecayDefinition] = None):
            A config for global weight decay

        uvm_host_mapped (bool = False): If True, allocate every UVM tensor
            using `malloc` + `cudaHostRegister`. Otherwise use
            `cudaMallocManaged`

        extra_optimizer_config Optional[UserEnabledConfigDefinition] = None):
            An extra config to enable certain modes for optimizer. These modes
            are not enabled by default.
            - `use_rowwise_bias_correction` is used in Adam to enable rowwise
                bias correction computation

        embedding_table_index_type (torch.dtype = torch.int64): The data type of
            the embedding table index tensor. Options are `torch.int32` and
            `torch.int64`

        embedding_table_offset_type (torch.dtype = torch.int64): The data type of
            the embedding table offset tensor. Options are `torch.int32` and
            `torch.int64`

        embedding_shard_info (Optional[List[Tuple[int, int, int, int]]] = None): the
            information about shard position and pre-sharded table size. If not set,
            the table is not sharded.
            (preshard_table_height, preshard_table_dim, height_offset, dim_offset)
    """

    embedding_specs: List[Tuple[int, int, EmbeddingLocation, ComputeDevice]]
    optimizer_args: invokers.lookup_args.OptimizerArgs
    lxu_cache_locations_list: List[Tensor]
    lxu_cache_locations_empty: Tensor
    timesteps_prefetched: List[int]
    record_cache_metrics: RecordCacheMetrics
    # pyre-fixme[13]: Attribute `uvm_cache_stats` is never initialized.
    uvm_cache_stats: torch.Tensor
    # pyre-fixme[13]: Attribute `local_uvm_cache_stats` is never initialized.
    local_uvm_cache_stats: torch.Tensor
    uuid: str
    # pyre-fixme[13]: Attribute `last_uvm_cache_print_state` is never initialized.
    last_uvm_cache_print_state: torch.Tensor
    _vbe_B_offsets: Optional[torch.Tensor]
    _vbe_max_B: int

    def __init__(  # noqa C901
        self,
        embedding_specs: List[
            Tuple[int, int, EmbeddingLocation, ComputeDevice]
        ],  # tuple of (rows, dims, placements, compute_devices)
        feature_table_map: Optional[List[int]] = None,  # [T]
        cache_algorithm: CacheAlgorithm = CacheAlgorithm.LRU,
        cache_load_factor: float = 0.2,
        cache_sets: int = 0,
        cache_reserved_memory: float = 0.0,
        cache_precision: Optional[SparseType] = None,
        weights_precision: SparseType = SparseType.FP32,
        output_dtype: SparseType = SparseType.FP32,
        enforce_hbm: bool = False,
        optimizer: OptimType = OptimType.EXACT_SGD,
        record_cache_metrics: Optional[RecordCacheMetrics] = None,
        gather_uvm_cache_stats: Optional[bool] = False,
        # General Optimizer args
        stochastic_rounding: bool = True,
        gradient_clipping: bool = False,
        max_gradient: float = 1.0,
        max_norm: float = 0.0,
        learning_rate: float = 0.01,
        eps: float = 1.0e-8,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        weight_decay_mode: WeightDecayMode = WeightDecayMode.NONE,
        eta: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        ensemble_mode: Optional[EnsembleModeDefinition] = None,
        emainplace_mode: Optional[EmainplaceModeDefinition] = None,
        counter_based_regularization: Optional[
            CounterBasedRegularizationDefinition
        ] = None,
        cowclip_regularization: Optional[CowClipDefinition] = None,
        pooling_mode: PoolingMode = PoolingMode.SUM,
        device: Optional[Union[str, int, torch.device]] = None,
        bounds_check_mode: BoundsCheckMode = BoundsCheckMode.WARNING,
        uvm_non_rowwise_momentum: bool = False,
        use_experimental_tbe: bool = False,
        prefetch_pipeline: bool = False,
        stats_reporter_config: Optional[TBEStatsReporterConfig] = None,
        table_names: Optional[List[str]] = None,
        optimizer_state_dtypes: Optional[Dict[str, SparseType]] = None,
        multipass_prefetch_config: Optional[MultiPassPrefetchConfig] = None,
        global_weight_decay: Optional[GlobalWeightDecayDefinition] = None,
        uvm_host_mapped: bool = False,
        extra_optimizer_config: Optional[UserEnabledConfigDefinition] = None,
        tbe_input_multiplexer_config: Optional[TBEInputMultiplexerConfig] = None,
        embedding_table_index_type: torch.dtype = torch.int64,
        embedding_table_offset_type: torch.dtype = torch.int64,
        embedding_shard_info: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> None:
        super(SplitTableBatchedEmbeddingBagsCodegen, self).__init__()
        self.uuid = str(uuid.uuid4())
        self.log("SplitTableBatchedEmbeddingBagsCodegen API: V2")
        self.log(f"SplitTableBatchedEmbeddingBagsCodegen Arguments: {locals()}")
        self.log(
            f"Feature Gates: {[(feature.name, feature.is_enabled()) for feature in FeatureGateName]}"
        )

        self.logging_table_name: str = self.get_table_name_for_logging(table_names)
        self.pooling_mode = pooling_mode
        self.is_nobag: bool = self.pooling_mode == PoolingMode.NONE

        # If environment variable is set, it overwrites the default bounds check mode.
        self.bounds_check_version: int = (
            2
            if self._feature_is_enabled(FeatureGateName.BOUNDS_CHECK_INDICES_V2)
            else get_bounds_check_version_for_platform()
        )
        self.bounds_check_mode_int: int = int(
            os.environ.get("FBGEMM_TBE_BOUNDS_CHECK_MODE", bounds_check_mode.value)
        )
        # Check if bounds_check_indices_v2 is enabled via the feature gate
        bounds_check_mode = BoundsCheckMode(self.bounds_check_mode_int)
        if bounds_check_mode.name.startswith("V2_"):
            self.bounds_check_version = 2
            if bounds_check_mode == BoundsCheckMode.V2_IGNORE:
                bounds_check_mode = BoundsCheckMode.IGNORE
            elif bounds_check_mode == BoundsCheckMode.V2_WARNING:
                bounds_check_mode = BoundsCheckMode.WARNING
            elif bounds_check_mode == BoundsCheckMode.V2_FATAL:
                bounds_check_mode = BoundsCheckMode.FATAL

        if bounds_check_mode not in (
            BoundsCheckMode.IGNORE,
            BoundsCheckMode.WARNING,
            BoundsCheckMode.FATAL,
            BoundsCheckMode.NONE,
        ):
            raise NotImplementedError(
                f"SplitTableBatchedEmbeddingBagsCodegen bounds_check_mode={bounds_check_mode} is not supported"
            )

        self.bounds_check_mode_int = bounds_check_mode.value

        self.log(
            f"SplitTableBatchedEmbeddingBagsCodegen bounds_check_mode={bounds_check_mode} bounds_check_version={self.bounds_check_version}"
        )

        self.weights_precision = weights_precision

        if torch.cuda.is_available() and torch.version.hip:
            # NOTE: It was discovered that FP16 cache precision caused a 500x
            # slowdown in performance of split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_kernel_warp_per_row_1
            # kernel on ROCm, so to work around this, we fix cache precision to
            # be FP32 always for the ROCm environment case.
            #
            # See:
            #   https://fb.workplace.com/groups/fbgemmusers/permalink/9438488366231860/
            cache_precision = SparseType.FP32
            self.log("Override cache_precision=SparseType.FP32 on ROCm")
        else:
            # NOTE: The changes from D65865527 are retained here until we can
            # test that the the hack also works for non-ROCm environments.
            cache_precision = (
                weights_precision if cache_precision is None else cache_precision
            )

        self.output_dtype: int = output_dtype.as_int()
        assert (
            not prefetch_pipeline or cache_algorithm == CacheAlgorithm.LRU
        ), "Only LRU cache policy supports prefetch_pipeline."
        self.prefetch_pipeline: bool = prefetch_pipeline
        self.lock_cache_line: bool = self.prefetch_pipeline
        self.use_uniq_cache_locations_bwd: bool = self.prefetch_pipeline
        self.multipass_prefetch_config: Optional[MultiPassPrefetchConfig] = (
            multipass_prefetch_config
        )

        if record_cache_metrics is not None:
            self.record_cache_metrics = record_cache_metrics
        else:
            self.record_cache_metrics = RecordCacheMetrics(False, False)

        if multipass_prefetch_config:
            assert (
                prefetch_pipeline
            ), "Multipass prefetch makes no sense in non-prefetch mode."
            assert (
                cache_algorithm == CacheAlgorithm.LRU
            ), "Multipass prefetch is only supported in LRU cache."
            assert (
                multipass_prefetch_config.num_passes > 0
            ), f"num_passes must be positive, get {multipass_prefetch_config.num_passes}"
            assert (
                multipass_prefetch_config.min_splitable_pass_size > 0
            ), f"min_splitable_pass_size must be positive, get {multipass_prefetch_config.min_splitable_pass_size}"
            assert (
                not self.record_cache_metrics.record_cache_miss_counter
                and not self.record_cache_metrics.record_tablewise_cache_miss
            ), "Unique cache miss counters are not accurate in multipass prefetch and therefore not supported"

        self.embedding_specs = embedding_specs
        (rows, dims, locations, compute_devices) = zip(*embedding_specs)
        T_ = len(self.embedding_specs)
        self.dims: List[int] = dims
        assert T_ > 0
        # mixed D is not supported by no bag kernels
        mixed_D = False
        D = self.dims[0]
        for d in self.dims:
            if d != D:
                mixed_D = True
                break
        if mixed_D:
            assert (
                self.pooling_mode != PoolingMode.NONE
            ), "Mixed dimension tables only supported for pooling tables."

        assert all(
            cd == compute_devices[0] for cd in compute_devices
        ), "Heterogenous compute_devices are NOT supported!"
        # Split TBE has different function schemas for CUDA and CPU.
        # For MTIA device type, it uses the CPU one.
        self.use_cpu: bool = (
            compute_devices[0] == ComputeDevice.CPU
            or compute_devices[0] == ComputeDevice.MTIA
        )

        assert not self.use_cpu or all(
            loc == EmbeddingLocation.HOST for loc in locations
        ), "ComputeDevice.CPU is only for EmbeddingLocation.HOST!"
        assert self.use_cpu or all(
            loc != EmbeddingLocation.HOST for loc in locations
        ), "EmbeddingLocation.HOST doesn't work for CUDA device!"
        if self.use_cpu or self.pooling_mode == PoolingMode.NONE:
            assert output_dtype in [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.BF16,
            ], "Fused pooled embedding quantization only supported for cuda."

        if optimizer == OptimType.NONE:
            assert all(
                loc == EmbeddingLocation.DEVICE for loc in locations
            ), "OptimType.NONE supports only EmbeddingLocation.DEVICE"
            assert all(
                cd == ComputeDevice.CUDA for cd in compute_devices
            ), "OptimType.NONE supports only ComputeDevice.CUDA"
            assert (
                not mixed_D
            ), "OptimType.NONE does not support mixed embedding dimension"

        if device is None:
            self.current_device: torch.device = (
                torch.device("cpu")
                if self.use_cpu
                else torch.device(torch.cuda.current_device())
            )
        elif isinstance(device, torch.device):
            self.current_device = device
        else:
            self.current_device = torch.device(device)

        # add placeholder require_grad param tensor to enable autograd with int8 weights
        self.placeholder_autograd_tensor = nn.Parameter(
            torch.zeros(0, device=self.current_device, dtype=torch.float)
        )

        self.gather_uvm_cache_stats = gather_uvm_cache_stats
        # Define the size of uvm cache stats as class variable
        # to make it work with torch jit script.
        self.uvm_cache_stats_size = 6
        # 0: N_calls, 1: N_requested_indices, 2: N_unique_indices, 3: N_unique_misses,
        # 4: N_conflict_unique_misses, 5: N_conflict_misses

        # Reporter to collect runtime performance stats bottom-up. Reporter may
        # do aggregation across TBEs and publish results per training batch.
        # Example of stats include UVM cache hit rate, table I/O size, etc.
        self.stats_reporter: Optional[TBEStatsReporter] = (
            stats_reporter_config.create_reporter() if stats_reporter_config else None
        )
        self._uvm_tensors_log: List[str] = []

        self.bwd_wait_prefetch_timer: Optional[AsyncSeriesTimer] = None
        self.prefetch_duration_timer: Optional[AsyncSeriesTimer] = None
        if self.stats_reporter:
            # When stats_reporter is present, we set up async series timer to
            # measure the GPU time per tracked event accordingly. Each of them
            # is attached to custom callback report function to report collected
            # duration with the corresponding event name.
            self.bwd_wait_prefetch_timer = AsyncSeriesTimer(
                functools.partial(
                    SplitTableBatchedEmbeddingBagsCodegen._report_duration,
                    self,
                    event_name="bwd_wait_for_prefetch",
                )
            )

            self.prefetch_duration_timer = AsyncSeriesTimer(
                functools.partial(
                    SplitTableBatchedEmbeddingBagsCodegen._report_duration,
                    self,
                    event_name="total_prefetch_duration",
                )
            )

        self.int8_emb_row_dim_offset: int = INT8_EMB_ROW_DIM_OFFSET

        self.feature_table_map: List[int] = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )

        if embedding_shard_info:
            (full_table_heights, full_table_dims, row_offset, col_offset) = zip(
                *embedding_shard_info
            )
        else:
            # Just assume the table is unsharded
            full_table_heights = rows
            full_table_dims = dims
            row_offset = [0] * len(rows)
            col_offset = [0] * len(rows)
        self.tbe_input_multiplexer: Optional[TBEInputMultiplexer] = (
            tbe_input_multiplexer_config.create_tbe_input_multiplexer(
                tbe_info=TBEInfo(
                    table_names=(
                        table_names
                        if table_names
                        else [f"table-{i}" for i in range(len(embedding_specs))]
                    ),
                    table_heights=rows,
                    tbe_uuid=self.uuid,
                    feature_table_map=self.feature_table_map,
                    table_dims=dims,
                    full_table_heights=full_table_heights,
                    full_table_dims=full_table_dims,
                    row_offset=row_offset,
                    col_offset=col_offset,
                )
            )
            if tbe_input_multiplexer_config is not None
            else None
        )
        T = len(self.feature_table_map)
        assert T_ <= T
        table_has_feature = [False] * T_
        for t in self.feature_table_map:
            table_has_feature[t] = True
        assert all(table_has_feature), "Each table must have at least one feature!"

        feature_dims = [dims[t] for t in self.feature_table_map]
        D_offsets = [0] + list(accumulate(feature_dims))
        self.total_D: int = D_offsets[-1]
        self.max_D: int = max(dims)
        cached_dims = [
            embedding_spec[1]
            for embedding_spec in embedding_specs
            if embedding_spec[2] == EmbeddingLocation.MANAGED_CACHING
        ]
        self.max_D_cache: int = max(cached_dims) if len(cached_dims) > 0 else 0

        self.register_buffer(
            "D_offsets",
            torch.tensor(D_offsets, device=self.current_device, dtype=torch.int32),
        )
        hash_size_cumsum = [0] + list(accumulate(rows))
        self.total_hash_size: int = int(hash_size_cumsum[-1])
        if self.total_hash_size == 0:
            self.total_hash_size_bits: int = 0
        else:
            self.total_hash_size_bits: int = int(log2(float(self.total_hash_size)) + 1)
        # The last element is to easily access # of rows of each table by
        # hash_size_cumsum[t + 1] - hash_size_cumsum[t]
        hash_size_cumsum = [hash_size_cumsum[t] for t in self.feature_table_map] + [
            self.total_hash_size
        ]
        self.register_buffer(
            "hash_size_cumsum",
            torch.tensor(
                hash_size_cumsum, device=self.current_device, dtype=torch.int64
            ),
        )

        self.register_buffer(
            "rows_per_table",
            torch.tensor(
                [rows[t] for t in self.feature_table_map],
                device=self.current_device,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "bounds_check_warning",
            torch.tensor([0], device=self.current_device, dtype=torch.int64),
        )
        # Required for VBE
        self.register_buffer(
            "feature_dims",
            torch.tensor(feature_dims, device="cpu", dtype=torch.int64),
        )
        (_info_B_num_bits, _info_B_mask) = torch.ops.fbgemm.get_infos_metadata(
            self.D_offsets,  # unused tensor
            1,  # max_B
            T,  # T
        )
        self.info_B_num_bits: int = _info_B_num_bits
        self.info_B_mask: int = _info_B_mask

        # A flag for indicating whether all embedding tables are placed in the
        # same locations
        self.use_homogeneous_placements: bool = all(
            loc == locations[0] for loc in locations
        )

        self.uvm_host_mapped = uvm_host_mapped

        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=weights_precision,
        )
        table_embedding_dtype = weights_precision.as_dtype()

        self._apply_split(
            weight_split,
            prefix="weights",
            # pyre-fixme[6]: For 3rd param expected `Type[Type[_dtype]]` but got
            #  `Type[_dtype]`.
            dtype=table_embedding_dtype,
            enforce_hbm=enforce_hbm,
            make_dev_param=optimizer == OptimType.NONE,
            dev_reshape=(-1, self.max_D) if optimizer == OptimType.NONE else None,
            uvm_host_mapped=self.uvm_host_mapped,
        )

        assert optimizer not in (
            OptimType.SGD,
            OptimType.ROWWISE_ADAGRAD,
        ), f"Optimizer {optimizer} is deprecated in the CPU + GPU modes."

        if self.use_cpu:
            # Construct optimizer states
            assert optimizer in (
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_SGD,
                OptimType.EMAINPLACE_ROWWISE_ADAGRAD,
            ), f"Optimizer {optimizer} is not supported in CPU mode."
        else:
            assert optimizer in (
                OptimType.ADAM,
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_SGD,
                OptimType.LAMB,
                OptimType.LARS_SGD,
                OptimType.PARTIAL_ROWWISE_ADAM,
                OptimType.PARTIAL_ROWWISE_LAMB,
                OptimType.ENSEMBLE_ROWWISE_ADAGRAD,
                OptimType.EMAINPLACE_ROWWISE_ADAGRAD,
                OptimType.NONE,
            ), f"Optimizer {optimizer} is not supported."

        self.stochastic_rounding = stochastic_rounding
        self.optimizer = optimizer

        self.weight_decay_mode = weight_decay_mode
        if (weight_decay_mode == WeightDecayMode.COUNTER) != (
            counter_based_regularization is not None
        ):
            raise AssertionError(
                "Need to set weight_decay_mode=WeightDecayMode.COUNTER together with valid counter_based_regularization"
            )
        if (weight_decay_mode == WeightDecayMode.COWCLIP) != (
            cowclip_regularization is not None
        ):
            raise AssertionError(
                "Need to set weight_decay_mode=WeightDecayMode.COWCLIP together with valid cowclip_regularization"
            )

        self._used_rowwise_adagrad_with_counter: bool = (
            optimizer == OptimType.EXACT_ROWWISE_ADAGRAD
            and (
                weight_decay_mode in (WeightDecayMode.COUNTER, WeightDecayMode.COWCLIP)
            )
        )

        if weight_decay_mode == WeightDecayMode.DECOUPLE_GLOBAL and (
            not optimizer == OptimType.EXACT_ROWWISE_ADAGRAD
            or global_weight_decay is None
        ):
            raise AssertionError(
                """weight_decay_mode=WeightDecayMode.DECOUPLE_GLOBAL is supported for
                optimizer=OptimType.EXACT_ROWWISE_ADAGRAD and global_weight_decay cannot be None.
                """
            )

        self._used_rowwise_adagrad_with_global_weight_decay: bool = (
            optimizer == OptimType.EXACT_ROWWISE_ADAGRAD
            and (weight_decay_mode == WeightDecayMode.DECOUPLE_GLOBAL)
        )
        self.log(
            f"Using global weight decay = {self._used_rowwise_adagrad_with_global_weight_decay}"
        )
        # Declare GWD params here to avoid torch.jit.script error
        if global_weight_decay is None:
            global_weight_decay = GlobalWeightDecayDefinition()

        self.gwd_start_iter: int = global_weight_decay.start_iter
        self.gwd_lower_bound: float = global_weight_decay.lower_bound

        if ensemble_mode is None:
            ensemble_mode = EnsembleModeDefinition()
        self._ensemble_mode: Dict[str, float] = {
            key: float(fval) for key, fval in ensemble_mode.__dict__.items()
        }

        if emainplace_mode is None:
            emainplace_mode = EmainplaceModeDefinition()
        self._emainplace_mode: Dict[str, float] = {
            key: float(fval) for key, fval in emainplace_mode.__dict__.items()
        }

        if counter_based_regularization is None:
            counter_based_regularization = CounterBasedRegularizationDefinition()
        if cowclip_regularization is None:
            cowclip_regularization = CowClipDefinition()
        self._max_counter_update_freq: int = -1
        # Extract parameters from CounterBasedRegularizationDefinition or CowClipDefinition
        # which are passed as entries for OptimizerArgs
        if self._used_rowwise_adagrad_with_counter:
            if self.weight_decay_mode == WeightDecayMode.COUNTER:
                self._max_counter_update_freq = (
                    counter_based_regularization.max_counter_update_freq
                )
                opt_arg_weight_decay_mode = (
                    counter_based_regularization.counter_weight_decay_mode
                )
                counter_halflife = counter_based_regularization.counter_halflife
            else:
                opt_arg_weight_decay_mode = (
                    cowclip_regularization.counter_weight_decay_mode
                )
                counter_halflife = cowclip_regularization.counter_halflife
        else:
            opt_arg_weight_decay_mode = weight_decay_mode
            # Default: -1, no decay applied, as a placeholder for OptimizerArgs
            # which should not be effective when CounterBasedRegularizationDefinition
            # and CowClipDefinition are not used
            counter_halflife = -1

        if extra_optimizer_config is None:
            extra_optimizer_config = UserEnabledConfigDefinition()
        self.use_rowwise_bias_correction: bool = (
            extra_optimizer_config.use_rowwise_bias_correction
        )
        self.use_writeback_bwd_prehook: bool = (
            extra_optimizer_config.use_writeback_bwd_prehook
        )
        self.log(f"self.extra_optimizer_config is {extra_optimizer_config}")
        if self.use_rowwise_bias_correction and not self.optimizer == OptimType.ADAM:
            raise AssertionError(
                "`use_rowwise_bias_correction` is only supported for OptimType.ADAM",
            )
        if self.use_writeback_bwd_prehook and not self.optimizer == OptimType.EXACT_SGD:
            raise AssertionError(
                "`use_writeback_bwd_prehook` is only supported for OptimType.EXACT_SGD",
            )

        self.learning_rate_tensor: torch.Tensor = torch.tensor(
            learning_rate, device=torch.device("cpu"), dtype=torch.float32
        )

        self.optimizer_args = invokers.lookup_args.OptimizerArgs(
            stochastic_rounding=stochastic_rounding,
            gradient_clipping=gradient_clipping,
            max_gradient=max_gradient,
            max_norm=max_norm,
            eps=eps,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            weight_decay_mode=opt_arg_weight_decay_mode.value,
            eta=eta,
            momentum=momentum,
            counter_halflife=counter_halflife,
            adjustment_iter=counter_based_regularization.adjustment_iter,
            adjustment_ub=counter_based_regularization.adjustment_ub,
            learning_rate_mode=counter_based_regularization.learning_rate_mode.value,
            grad_sum_decay=counter_based_regularization.grad_sum_decay.value,
            tail_id_threshold=counter_based_regularization.tail_id_threshold.val,
            is_tail_id_thresh_ratio=int(
                counter_based_regularization.tail_id_threshold.is_ratio
            ),
            total_hash_size=self.total_hash_size,
            weight_norm_coefficient=cowclip_regularization.weight_norm_coefficient,
            lower_bound=cowclip_regularization.lower_bound,
            regularization_mode=weight_decay_mode.value,
            use_rowwise_bias_correction=self.use_rowwise_bias_correction,
        )

        if optimizer != OptimType.NONE:
            assert (
                optimizer
                in (OptimType.PARTIAL_ROWWISE_ADAM, OptimType.ENSEMBLE_ROWWISE_ADAGRAD)
                or optimizer_state_dtypes is None
            ), "optimizer_state_dtypes option is only supported for OptimType.PARTIAL_ROWWISE_ADAM and OptimType.ENSEMBLE_ROWWISE_ADAGRAD"
            if optimizer in (OptimType.EXACT_SGD,):
                # NOTE: make TorchScript work!
                self._register_nonpersistent_buffers("momentum1")
            else:
                momentum1_dtype = (
                    torch.float32
                    if (
                        optimizer_state_dtypes is None
                        or "momentum1" not in optimizer_state_dtypes
                        or optimizer == OptimType.ENSEMBLE_ROWWISE_ADAGRAD
                    )
                    else optimizer_state_dtypes["momentum1"].as_dtype()
                )
                rowwise = optimizer in [
                    OptimType.EXACT_ROWWISE_ADAGRAD,
                    OptimType.ENSEMBLE_ROWWISE_ADAGRAD,
                    OptimType.EMAINPLACE_ROWWISE_ADAGRAD,
                ]
                self._apply_split(
                    construct_split_state(
                        embedding_specs,
                        rowwise=rowwise,
                        cacheable=False,
                        placement=(
                            EmbeddingLocation.MANAGED
                            if ((not rowwise) and uvm_non_rowwise_momentum)
                            else None
                        ),
                    ),
                    prefix="momentum1",
                    # pyre-fixme[6]: Expected `Type[Type[torch._dtype]]` for 3rd param
                    #  but got `Type[torch.float32]`.
                    dtype=momentum1_dtype,
                    enforce_hbm=enforce_hbm,
                    uvm_host_mapped=self.uvm_host_mapped,
                )
            if optimizer in (
                OptimType.ADAM,
                OptimType.PARTIAL_ROWWISE_ADAM,
                OptimType.LAMB,
                OptimType.PARTIAL_ROWWISE_LAMB,
                OptimType.ENSEMBLE_ROWWISE_ADAGRAD,
            ):
                rowwise = optimizer in (
                    OptimType.PARTIAL_ROWWISE_ADAM,
                    OptimType.PARTIAL_ROWWISE_LAMB,
                )
                momentum2_dtype = (
                    torch.float32
                    if (
                        optimizer_state_dtypes is None
                        or "momentum2" not in optimizer_state_dtypes
                    )
                    else optimizer_state_dtypes["momentum2"].as_dtype()
                )
                self._apply_split(
                    construct_split_state(
                        embedding_specs,
                        rowwise=rowwise,
                        cacheable=False,
                        placement=(
                            EmbeddingLocation.MANAGED
                            if ((not rowwise) and uvm_non_rowwise_momentum)
                            else None
                        ),
                    ),
                    prefix="momentum2",
                    # pyre-fixme[6]: Expected `Type[Type[torch._dtype]]` for 3rd param
                    #  but got `Type[torch.float32]`.
                    dtype=momentum2_dtype,
                    uvm_host_mapped=self.uvm_host_mapped,
                )
            else:
                # NOTE: make TorchScript work!
                self._register_nonpersistent_buffers("momentum2")
            if self._used_rowwise_adagrad_with_counter:
                self._apply_split(
                    construct_split_state(
                        embedding_specs,
                        rowwise=True,
                        cacheable=False,
                    ),
                    prefix="prev_iter",
                    # TODO: ideally we should use int64 to track iter but it failed to compile.
                    # It may be related to low precision training code. Currently using float32
                    # as a workaround while investigating the issue.
                    # pyre-fixme[6]: Expected `Type[Type[torch._dtype]]` for 3rd param
                    #  but got `Type[torch.float32]`.
                    dtype=torch.float32,
                    uvm_host_mapped=self.uvm_host_mapped,
                )
                self._apply_split(
                    construct_split_state(
                        embedding_specs,
                        rowwise=True,
                        cacheable=False,
                    ),
                    prefix="row_counter",
                    # pyre-fixme[6]: Expected `Type[Type[torch._dtype]]` for 3rd param
                    #  but got `Type[torch.float32]`.
                    dtype=torch.float32,
                    uvm_host_mapped=self.uvm_host_mapped,
                )
                self.register_buffer(
                    "max_counter", torch.tensor([1], dtype=torch.float32)
                )
            elif self._used_rowwise_adagrad_with_global_weight_decay:
                self._apply_split(
                    construct_split_state(
                        embedding_specs,
                        rowwise=True,
                        cacheable=False,
                    ),
                    prefix="prev_iter",
                    # TODO: ideally we should use int64 to track iter but it failed to compile.
                    # It may be related to low precision training code. Currently using float32
                    # as a workaround while investigating the issue.
                    # pyre-fixme[6]: Expected `Type[Type[torch._dtype]]` for 3rd param
                    #  but got `Type[torch.float32]`.
                    dtype=torch.float32,
                    uvm_host_mapped=self.uvm_host_mapped,
                )
                self._register_nonpersistent_buffers("row_counter")
                self.register_buffer(
                    "max_counter",
                    torch.ones(1, dtype=torch.float32, device=self.current_device),
                    persistent=False,
                )
            elif optimizer == OptimType.ADAM and self.use_rowwise_bias_correction:
                self._apply_split(
                    construct_split_state(
                        embedding_specs,
                        rowwise=True,
                        cacheable=False,
                    ),
                    prefix="row_counter",
                    # pyre-fixme[6]: Expected `Type[Type[torch._dtype]]` for 3rd param
                    #  but got `Type[torch.float32]`.
                    dtype=torch.float32,
                    uvm_host_mapped=self.uvm_host_mapped,
                )
            else:
                self._register_nonpersistent_buffers("prev_iter")
                self._register_nonpersistent_buffers("row_counter")
                self.register_buffer(
                    "max_counter",
                    torch.ones(1, dtype=torch.float32, device=self.current_device),
                    persistent=False,
                )
            if (
                optimizer
                in (
                    OptimType.ADAM,
                    OptimType.LAMB,
                    OptimType.PARTIAL_ROWWISE_ADAM,
                    OptimType.PARTIAL_ROWWISE_LAMB,
                    OptimType.ENSEMBLE_ROWWISE_ADAGRAD,
                    OptimType.EMAINPLACE_ROWWISE_ADAGRAD,
                )
                or self._used_rowwise_adagrad_with_global_weight_decay
            ):
                self.register_buffer(
                    "iter",
                    torch.zeros(1, dtype=torch.int64, device=self.current_device),
                )
            else:
                self.register_buffer(
                    "iter",
                    torch.zeros(1, dtype=torch.int64, device=self.current_device),
                    persistent=False,
                )

        self.iter_cpu: torch.Tensor = torch.zeros(1, dtype=torch.int64, device="cpu")

        cache_state = construct_cache_state(rows, locations, self.feature_table_map)

        # Add table-wise cache miss counter
        if self.record_cache_metrics.record_tablewise_cache_miss:
            num_tables = len(cache_state.cache_hash_size_cumsum) - 1
            self.register_buffer(
                "table_wise_cache_miss",
                torch.zeros(
                    num_tables,
                    device=self.current_device,
                    dtype=torch.int64,
                ),
            )
        # NOTE: make TorchScript work!
        else:
            self.register_buffer(
                "table_wise_cache_miss",
                torch.zeros(
                    0,
                    device=self.current_device,
                    dtype=torch.int64,
                ),
            )

        self._apply_cache_state(
            cache_state,
            cache_algorithm,
            cache_load_factor,
            cache_sets,
            cache_reserved_memory,
            cache_precision,
        )

        self.log(f"Contents: {table_names}")
        self.log(
            f"Using fused {optimizer} with optimizer_args={self.optimizer_args if optimizer != OptimType.NONE else None}"
        )
        self.log(
            f"Using rowwise_adagrad_with_counter={self._used_rowwise_adagrad_with_counter}"
        )

        self.step = 0
        self.last_reported_step = 0
        self.last_reported_uvm_stats: List[float] = []

        # Check whether to use TBE v2
        is_experimental = False
        if use_experimental_tbe:
            is_experimental = True
            self.log("use_experimental_tbe is set to True; Using experimental TBE")

        elif int(os.environ.get("FBGEMM_EXPERIMENTAL_TBE", "0")) == 1:
            # Keep the old feature enablement mechanism to ensure no negative impact on models that have already adopted TBE v2
            is_experimental = True
            self.log("FBGEMM_EXPERIMENTAL_TBE is set to True; Using experimental TBE")

        # NOTE: Keep this disabled for now until the backend lands into Pyper
        # elif FeatureGateName.TBE_V2.is_enabled():
        #     is_experimental = True
        #     self.log("TBE_V2 Knob is set to True; Using experimental TBE")

        self.is_experimental: bool = is_experimental

        # Get a debug function pointer
        self._debug_print_input_stats: Callable[..., None] = (
            self._debug_print_input_stats_factory()
        )

        if optimizer == OptimType.EXACT_SGD and self.use_writeback_bwd_prehook:
            # Register writeback hook for Exact_SGD optimizer
            self.log(
                "SplitTableBatchedEmbeddingBagsCodegen:  use_writeback_bwd_prehook is enabled."
            )
            # pyre-fixme[6]: Expected `typing.Callable[[Module, Union[Tensor, typing.Tuple[Tensor, ...]]], Union[None, Tensor, typing.Tuple[Tensor, ...]]]`
            self.register_full_backward_pre_hook(self.writeback_hook)

        if embedding_table_index_type not in [torch.int32, torch.int64]:
            raise ValueError(
                f"embedding_table_index_type must be torch.int32 or torch.int64, but got {embedding_table_index_type}"
            )
        self.embedding_table_index_type: torch.dtype = embedding_table_index_type
        if embedding_table_offset_type not in [torch.int32, torch.int64]:
            raise ValueError(
                f"embedding_table_offset_type must be torch.int32 or torch.int64, but got {embedding_table_offset_type}"
            )
        self.embedding_table_offset_type: torch.dtype = embedding_table_offset_type

    @torch.jit.ignore
    def log(self, msg: str) -> None:
        """
        Log with TBE id prefix to distinguish between multiple TBE instances
        per process

        Args:
            msg (str): The message to print

        Returns:
            None
        """
        logging.info(f"[TBE={self.uuid}] {msg}")

    def _register_nonpersistent_buffers(self, prefix: str) -> None:
        # NOTE: make TorchScript work!
        self.register_buffer(
            f"{prefix}_dev",
            torch.zeros(1, dtype=torch.int64, device=self.current_device),
            persistent=False,
        )
        self.register_buffer(
            f"{prefix}_host",
            torch.zeros(1, dtype=torch.int64, device=self.current_device),
            persistent=False,
        )
        self.register_buffer(
            f"{prefix}_uvm",
            torch.zeros(1, dtype=torch.int64, device=self.current_device),
            persistent=False,
        )
        self.register_buffer(
            f"{prefix}_placements",
            torch.zeros(1, dtype=torch.int64, device=self.current_device),
            persistent=False,
        )
        self.register_buffer(
            f"{prefix}_offsets",
            torch.zeros(1, dtype=torch.int64, device=self.current_device),
            persistent=False,
        )

    @staticmethod
    def get_table_name_for_logging(table_names: Optional[List[str]]) -> str:
        """
        Given a list of all table names in the TBE, generate a string to
        represent them in logging. If there is more than one table, this method
        will count them than list them.

        Args:
            table_names (Optional[List[str]]): A list of table anmes in TBE

        Returns:
            A string that represents tables in logging
        """
        if table_names is None:
            return "<Unknown>"
        # Do this because sometimes multiple shards of the same table could appear
        # in one TBE.
        table_name_set = set(table_names)
        if len(table_name_set) == 1:
            return next(iter(table_name_set))
        return f"<{len(table_name_set)} tables>"

    @staticmethod
    def get_prefetch_passes(
        multipass_prefetch_config: Optional[MultiPassPrefetchConfig],
        input_tensor: Tensor,
        output_tensor: Tensor,
    ) -> List[Tuple[Tensor, Tensor, int]]:
        """
        Given inputs (the indices to forward), partition the input and output
        into smaller chunks and return them as a list of tuples
        (input[start_idx:end_idx], output[start_idx:end_idx], start_idx).

        The caller must guarantee that input and output have non-zero dimension
        0. The returned segments are guaranteed to completely and
        non-overlappingly cover the input tensor.

        In non-multipass-prefetch mode, it returns the input/output tensor
        itself.

        Args:
            multipass_prefetch_config (Optional[MultiPassPrefetchConfig]):
                A config for multi-pass cache prefetch. If None, multi-pass
                prefetch is not used.

            input_tensor (Tensor): The input tensor to be partitioned

            output_tensor (Tensor): The output tensor to be partitioned

        Returns:
            A list of partitioned inputs and outputs (List[Tuple[Tensor,
                Tensor, int]])
        """
        if multipass_prefetch_config is None:
            return [(input_tensor, output_tensor, 0)]
        mpp_config: MultiPassPrefetchConfig = multipass_prefetch_config

        N = input_tensor.size(0)
        if N <= mpp_config.num_passes or mpp_config.num_passes == 1:
            # One row per pass, just don't split
            return [(input_tensor, output_tensor, 0)]

        pass_size: int = max(
            (N + mpp_config.num_passes - 1) // mpp_config.num_passes,
            mpp_config.min_splitable_pass_size,
        )

        return list(
            zip(
                torch.split(input_tensor, pass_size),
                torch.split(output_tensor, pass_size),
                range(0, N, pass_size),
            )
        )

    def get_states(self, prefix: str) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Get a state of a given tensor (`prefix`)

        Args:
            prefix (str): A prefix of the state to obtain

        Returns:
            A tuple of tensors corresponding to the obtained state containing

            (1) A GPU state tensor

            (2) A CPU state tensor

            (3) A UVM state tensor

            (4) A placement tensor - containing placements of embedding tables
                (torch.int32_t tensor). (0 = DEVICE, 1 = MANAGED, 2 =
                MANAGED_CACHING, 3 = HOST, 4 = MTIA)

            (5) An offset tensor - containing the relative positions of
                embedding tables in the corresponding state tensor (GPU, CPU,
                or UVM state tensor)
        """
        if not hasattr(self, f"{prefix}_physical_placements"):
            raise DoesNotHavePrefix()
        dev_param = getattr(self, f"{prefix}_dev")
        host_param = getattr(self, f"{prefix}_host")
        uvm_param = getattr(self, f"{prefix}_uvm")
        placements = getattr(self, f"{prefix}_physical_placements")
        offsets = getattr(self, f"{prefix}_physical_offsets")
        return (
            dev_param,
            host_param,
            uvm_param,
            torch.tensor(placements, dtype=torch.int32),
            torch.tensor(offsets, dtype=torch.int64),
        )

    def get_all_states(self) -> List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """
        Get all states in the TBE (`weights`, `momentum1`, `momentum2`,
        `prev_iter`, and `row_counter`)

        Returns:
            A list of states. Each state is a tuple of tensors (GPU state
            tensor, CPU state tensor, UVM state tensor, placement tensor and
            offset tensor)
        """
        all_states = []
        for prefix in ["weights", "momentum1", "momentum2", "prev_iter", "row_counter"]:
            try:
                all_states.append(self.get_states(prefix))
            except DoesNotHavePrefix:
                pass
        return all_states

    @torch.jit.export
    def get_cache_miss_counter(self) -> Tensor:
        """
        Get the cache miss counter. `cache_miss_counter` contains two items:

        (1) `cache_miss_forward_count` which records the total number of
            forwards which has at least one cache miss

        (2) `unique_cache_miss_count` which records to total number of unique
            (dedup) cache misses

        Returns:
            The cache miss counter
        """
        # pyre-fixme[7]: Expected `Tensor` but got `Union[Module, Tensor]`.
        return self.cache_miss_counter

    @torch.jit.export
    def get_table_wise_cache_miss(self) -> Tensor:
        """
        Get the table-wise cache miss tensor. `table_wise_cache_miss` contains
        all the cache miss count for each table in this embedding table object:

        Returns:
            The table-wise cache miss tensor
        """
        return self.table_wise_cache_miss

    # The callback function for AsyncTimer to record duration to different event
    def _report_duration(
        self,
        it_step: int,
        dur_ms: float,
        event_name: str,
    ) -> None:
        assert (
            self.stats_reporter
        ), "We should not be here. AsyncTimer only happens with reporter present."
        self.stats_reporter.report_duration(
            iteration_step=it_step,
            event_name=event_name,
            duration_ms=dur_ms,
            embedding_id=self.logging_table_name,
            tbe_id=self.uuid,
        )

    @torch.jit.ignore
    def _report_tbe_mem_usage(
        self,
    ) -> None:
        if self.stats_reporter is None:
            return

        stats_reporter: TBEStatsReporter = self.stats_reporter
        if not stats_reporter.should_report(self.step):
            return

        total_mem_usage = sum(
            param.numel() * param.element_size() for param in self.parameters()
        ) + sum(buffer.numel() * buffer.element_size() for buffer in self.buffers())
        if self.use_cpu:
            total_hbm_usage = 0
            total_uvm_usage = total_mem_usage
        else:
            # hbm usage is total usage minus uvm usage
            total_uvm_usage = sum(
                getattr(self, tensor_name).numel()
                * getattr(self, tensor_name).element_size()
                for tensor_name in self._uvm_tensors_log
                if hasattr(self, tensor_name)
            )
            total_hbm_usage = total_mem_usage - total_uvm_usage

        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name="tbe.total_hbm_usage",
            data_bytes=total_hbm_usage,
            embedding_id=self.logging_table_name,
            tbe_id=self.uuid,
        )
        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name="tbe.total_uvm_usage",
            data_bytes=total_uvm_usage,
            embedding_id=self.logging_table_name,
            tbe_id=self.uuid,
        )

    @torch.jit.ignore
    def _report_io_size_count(self, event: str, data: Tensor) -> Tensor:
        if self.stats_reporter is None:
            return data
        stats_reporter: TBEStatsReporter = self.stats_reporter
        if stats_reporter.should_report(self.step):
            stats_reporter.report_data_amount(
                iteration_step=self.step,
                event_name=f"tbe.{event}_size",
                data_bytes=data.element_size() * data.numel(),
                embedding_id=self.logging_table_name,
                tbe_id=self.uuid,
            )
            stats_reporter.report_data_amount(
                iteration_step=self.step,
                event_name=f"tbe.{event}_count",
                data_bytes=data.numel(),
                embedding_id=self.logging_table_name,
                tbe_id=self.uuid,
            )
        return data

    @torch.jit.ignore
    def _generate_vbe_metadata(
        self,
        offsets: Tensor,
        batch_size_per_feature_per_rank: Optional[List[List[int]]],
    ) -> invokers.lookup_args.VBEMetadata:
        # Blocking D2H copy, but only runs at first call
        self.feature_dims = self.feature_dims.cpu()
        if batch_size_per_feature_per_rank is not None:
            assert self.optimizer in (
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_SGD,
                OptimType.ENSEMBLE_ROWWISE_ADAGRAD,
                OptimType.EMAINPLACE_ROWWISE_ADAGRAD,
                OptimType.NONE,
                OptimType.ADAM,
            ), (
                "Variable batch size TBE support is enabled for "
                "OptimType.EXACT_ROWWISE_ADAGRAD,EXACT_SGD, "
                "ENSEMBLE_ROWWISE_ADAGRAD, NONE, and ADAM only"
            )
        return generate_vbe_metadata(
            offsets,
            batch_size_per_feature_per_rank,
            self.pooling_mode,
            self.feature_dims,
            self.current_device,
        )

    @torch.jit.ignore
    def _feature_is_enabled(self, feature: FeatureGateName) -> bool:
        # Define proxy method so that it can be marked with @torch.jit.ignore
        # This allows models using this class to compile correctly
        return FeatureGate.is_enabled(feature)

    def writeback_update_gradient(
        self, indices: torch.Tensor, offsets: torch.Tensor, grad: Tensor
    ) -> Tensor:
        if indices.numel() == 0:
            return grad[0]
        num_of_tables = len(set(self.feature_table_map))
        assert num_of_tables * indices.max() < torch.iinfo(indices.dtype).max
        batch_size = offsets.shape[0] // num_of_tables
        max_indices = indices.max()
        non_empty_index = (offsets[1:] - offsets[:-1]).nonzero().flatten()
        # disable dedup across different table
        indices = ((offsets[non_empty_index]) // batch_size) * (
            1 + max_indices
        ) + indices
        grad = grad[0]
        _, idx, counts = torch.unique(
            indices, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]).to(indices.device), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]
        mask = torch.zeros_like(grad, device=grad.device)
        original_index = non_empty_index[first_indicies]

        mask[original_index] = grad[original_index]
        return mask

    # pyre-fixme[2]: For 1st argument expected not ANY
    def writeback_hook(self, module: Any, grad: Tensor) -> Tuple[Tensor]:
        indices = self._indices
        offsets = self._offsets

        return (self.writeback_update_gradient(indices, offsets, grad),)

    def forward(  # noqa: C901
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        total_unique_indices: Optional[int] = None,
    ) -> Tensor:
        """
        The forward pass function that

        (1) Performs input bound checking

        (2) Generates necessary variable batch size embedding (VBE) metadata (if
            VBE is used)

        (3) Prefetches data from UVM to cache (if
            `EmbeddingLocation.MANAGED_CACHING` is used and the user has not
            explicitly prefetched data)

        (4) Performs the embedding table lookup by invoking a corresponding
            Autograd function (based on the chosen optimizer)

        Args:
            indices (Tensor): A 1D-tensor that contains indices to be looked up
                from all embedding table

            offsets (Tensor): A 1D-tensor that conatins offsets of indices.
                Shape `(B * T + 1)` where `B` = batch size and `T` = the number
                of features.  `offsets[t * B + b + 1] - offsets[t * B + b]` is
                the length of bag `b` of feature `t`

            per_sample_weights (Optional[Tensor]): An optional 1D-float-tensor that
                contains per sample weights. If None, **unweighted** embedding
                lookup will be perform. Otherwise, **weighted** will be used. The
                length of this tensor must be the same as the length of the
                `indices` tensor.  The value of `per_sample_weights[i]` will be
                used to multiply with every element in the looked up row
                `indices[i]`, where `0 <= i < len(per_sample_weights)`.

            feature_requires_grad (Optional[Tensor]): An optional 1D-tensor for
                indicating if `per_sample_weights` requires gradient. The
                length of the tensor must be equal to the number of features

            batch_size_per_feature_per_rank (Optional[List[List[int]]]): An
                optional 2D-tensor that contains batch sizes for every rank and
                every feature. If None, TBE assumes that **every feature has the
                same batch size** and computes the batch size from the `offsets`
                shape. Otherwise, TBE assumes that different features can have
                different batch sizes and uses the **variable batch size
                embedding look up mode (VBE)**. Shape (number of features,
                number of ranks). `batch_size_per_feature_per_rank[f][r]`
                represents the batch size of feature `f` and rank `r`

            total_unique_indices (Optional[int]): An optional integer that
                represents the total number of unique indices. This value must
                be set when using `OptimType.NONE`. This is because TBE
                requires this information for allocating the weight gradient
                tensor in the backward pass.

        Returns:
            A 2D-tensor containing looked up data. Shape `(B, total_D)` where `B` =
            batch size and `total_D` = the sum of all embedding dimensions in the
            table

        Example:

            >>> import torch
            >>>
            >>> from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
            >>>    EmbeddingLocation,
            >>> )
            >>> from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            >>>    SplitTableBatchedEmbeddingBagsCodegen,
            >>>    ComputeDevice,
            >>> )
            >>>
            >>> # Two tables
            >>> embedding_specs = [
            >>>     (3, 8, EmbeddingLocation.DEVICE, ComputeDevice.CUDA),
            >>>     (5, 4, EmbeddingLocation.MANAGED, ComputeDevice.CUDA)
            >>> ]
            >>>
            >>> tbe = SplitTableBatchedEmbeddingBagsCodegen(embedding_specs)
            >>> tbe.init_embedding_weights_uniform(-1, 1)
            >>>
            >>> print(tbe.split_embedding_weights())
            [tensor([[-0.9426,  0.7046,  0.4214, -0.0419,  0.1331, -0.7856, -0.8124, -0.2021],
                    [-0.5771,  0.5911, -0.7792, -0.1068, -0.6203,  0.4813, -0.1677,  0.4790],
                    [-0.5587, -0.0941,  0.5754,  0.3475, -0.8952, -0.1964,  0.0810, -0.4174]],
                   device='cuda:0'), tensor([[-0.2513, -0.4039, -0.3775,  0.3273],
                    [-0.5399, -0.0229, -0.1455, -0.8770],
                    [-0.9520,  0.4593, -0.7169,  0.6307],
                    [-0.1765,  0.8757,  0.8614,  0.2051],
                    [-0.0603, -0.9980, -0.7958, -0.5826]], device='cuda:0')]


            >>> # Batch size = 3
            >>> indices = torch.tensor([0, 1, 2, 0, 1, 2, 0, 3, 1, 4, 2, 0, 0],
            >>>                        device="cuda",
            >>>                        dtype=torch.long)
            >>> offsets = torch.tensor([0, 2, 5, 7, 9, 12, 13],
            >>>                        device="cuda",
            >>>                        dtype=torch.long)
            >>>
            >>> output = tbe(indices, offsets)
            >>>
            >>> # Batch size = 3, total embedding dimension = 12
            >>> print(output.shape)
            torch.Size([3, 12])

            >>> print(output)
            tensor([[-1.5197,  1.2957, -0.3578, -0.1487, -0.4873, -0.3044, -0.9801,  0.2769,
                     -0.7164,  0.8528,  0.7159, -0.6719],
                    [-2.0784,  1.2016,  0.2176,  0.1988, -1.3825, -0.5008, -0.8991, -0.1405,
                     -1.2637, -0.9427, -1.8902,  0.3754],
                    [-1.5013,  0.6105,  0.9968,  0.3057, -0.7621, -0.9821, -0.7314, -0.6195,
                     -0.2513, -0.4039, -0.3775,  0.3273]], device='cuda:0',
                   grad_fn=<CppNode<SplitLookupFunction_sgd_Op>>)

        """
        (
            indices,
            offsets,
            per_sample_weights,
            vbe_metadata,
        ) = self.prepare_inputs(
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
            force_cast_input_types=True,
            prefetch_pipeline=False,
        )

        # Print input stats if enable (for debugging purpose only)
        self._debug_print_input_stats(indices, offsets, per_sample_weights)

        if not is_torchdynamo_compiling():
            # Mutations of nn.Module attr forces dynamo restart of Analysis which increases compilation time

            # Storing tensors for linear_cache_indices recomputation
            self._indices = indices
            self._offsets = offsets
            self._vbe_B_offsets = vbe_metadata.B_offsets
            self._vbe_max_B = vbe_metadata.max_B

            self.step += 1
            self._report_io_size_count("fwd_input", indices)
            self._report_tbe_mem_usage()

            if self.tbe_input_multiplexer is not None:
                tbe_input_multiplexer: TBEInputMultiplexer = self.tbe_input_multiplexer
                if tbe_input_multiplexer.should_run(self.step):
                    tbe_input_multiplexer.run(
                        tbe_input_info=TBEInputInfo(
                            indices, offsets, batch_size_per_feature_per_rank
                        )
                    )

        if len(self.timesteps_prefetched) == 0:
            # In forward, we don't enable multi-pass prefetch as we want the process
            # to be as fast as possible and memory usage doesn't matter (will be recycled
            # by dense fwd/bwd)
            self._prefetch(
                indices, offsets, vbe_metadata, multipass_prefetch_config=None
            )

        if len(self.timesteps_prefetched) > 0:
            self.timesteps_prefetched.pop(0)

        self.lxu_cache_locations = (
            self.lxu_cache_locations_empty
            if len(self.lxu_cache_locations_list) == 0
            else self.lxu_cache_locations_list.pop(0)
        )
        common_args = invokers.lookup_args.CommonArgs(
            placeholder_autograd_tensor=self.placeholder_autograd_tensor,
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            dev_weights=self.weights_dev,
            # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            host_weights=self.weights_host,
            # pyre-fixme[6]: For 4th argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            uvm_weights=self.weights_uvm,
            # pyre-fixme[6]: For 5th argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            lxu_cache_weights=self.lxu_cache_weights,
            # pyre-fixme[6]: For 6th argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            weights_placements=self.weights_placements,
            # pyre-fixme[6]: For 7th argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            weights_offsets=self.weights_offsets,
            D_offsets=self.D_offsets,
            total_D=self.total_D,
            max_D=self.max_D,
            hash_size_cumsum=self.hash_size_cumsum,
            total_hash_size_bits=self.total_hash_size_bits,
            indices=indices,
            offsets=offsets,
            pooling_mode=self.pooling_mode,
            indice_weights=per_sample_weights,
            feature_requires_grad=feature_requires_grad,
            lxu_cache_locations=self.lxu_cache_locations,
            # Pass the local_uvm_cache_stats bc only that information is
            # relevant for the current iteration
            uvm_cache_stats=(
                self.local_uvm_cache_stats
                if (
                    self.gather_uvm_cache_stats
                    # Unique conflict misses are only collected when using CacheAlgorithm.LRU
                    and self.cache_algorithm == CacheAlgorithm.LRU
                )
                else None
            ),
            output_dtype=self.output_dtype,
            vbe_metadata=vbe_metadata,
            is_experimental=self.is_experimental,
            use_uniq_cache_locations_bwd=self.use_uniq_cache_locations_bwd,
            use_homogeneous_placements=self.use_homogeneous_placements,
            learning_rate_tensor=self.learning_rate_tensor,
            info_B_num_bits=self.info_B_num_bits,
            info_B_mask=self.info_B_mask,
        )

        if self.optimizer == OptimType.NONE:
            assert (
                total_unique_indices is not None
                and total_unique_indices <= indices.numel()
            ), f"OptimType.NONE requires total_unique_indices. Please pass it or check the value (total_unique_indices = {total_unique_indices})"
            return self._report_io_size_count(
                "fwd_output",
                invokers.lookup_none.invoke(
                    common_args, self.optimizer_args, total_unique_indices
                ),
            )
        elif self.optimizer == OptimType.EXACT_SGD:
            return self._report_io_size_count(
                "fwd_output",
                invokers.lookup_sgd.invoke(common_args, self.optimizer_args),
            )

        momentum1 = invokers.lookup_args.Momentum(
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            dev=self.momentum1_dev,
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            host=self.momentum1_host,
            # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            uvm=self.momentum1_uvm,
            # pyre-fixme[6]: For 4th argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            offsets=self.momentum1_offsets,
            # pyre-fixme[6]: For 5th argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            placements=self.momentum1_placements,
        )

        if self.optimizer == OptimType.LARS_SGD:
            return self._report_io_size_count(
                "fwd_output",
                invokers.lookup_lars_sgd.invoke(
                    common_args, self.optimizer_args, momentum1
                ),
            )
        if self.optimizer == OptimType.EXACT_ADAGRAD:
            return self._report_io_size_count(
                "fwd_output",
                invokers.lookup_adagrad.invoke(
                    common_args, self.optimizer_args, momentum1
                ),
            )

        momentum2 = invokers.lookup_args.Momentum(
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            dev=self.momentum2_dev,
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            host=self.momentum2_host,
            # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            uvm=self.momentum2_uvm,
            # pyre-fixme[6]: For 4th argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            offsets=self.momentum2_offsets,
            # pyre-fixme[6]: For 5th argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            placements=self.momentum2_placements,
        )

        # Although self.iter_cpu is created on CPU. It might be transferred to
        # GPU by the user. So, we need to transfer it to CPU explicitly. This
        # should be done only once.
        self.iter_cpu = self.iter_cpu.cpu()

        # Sync with loaded state
        if (
            not is_torchdynamo_compiling()
        ):  # wrap to make it compatible with PT2 compile
            if self.iter_cpu.item() == 0:
                self.iter_cpu.fill_(self.iter.cpu().item())
        # Increment the iteration counter
        iter_int = int(self.iter_cpu.add_(1).item())  # used for local computation
        self.iter.add_(1)  # used for checkpointing

        row_counter = invokers.lookup_args.Momentum(
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            dev=self.row_counter_dev,
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            host=self.row_counter_host,
            # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            uvm=self.row_counter_uvm,
            # pyre-fixme[6]: For 4th argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            offsets=self.row_counter_offsets,
            # pyre-fixme[6]: For 5th argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            placements=self.row_counter_placements,
        )

        if self.optimizer == OptimType.ADAM:
            return self._report_io_size_count(
                "fwd_output",
                invokers.lookup_adam.invoke(
                    common_args,
                    self.optimizer_args,
                    momentum1,
                    momentum2,
                    iter_int,
                    row_counter=(
                        row_counter if self.use_rowwise_bias_correction else None
                    ),
                ),
            )
        if self.optimizer == OptimType.PARTIAL_ROWWISE_ADAM:
            return self._report_io_size_count(
                "fwd_output",
                invokers.lookup_partial_rowwise_adam.invoke(
                    common_args,
                    self.optimizer_args,
                    momentum1,
                    momentum2,
                    iter_int,
                ),
            )
        if self.optimizer == OptimType.LAMB:
            return self._report_io_size_count(
                "fwd_output",
                invokers.lookup_lamb.invoke(
                    common_args,
                    self.optimizer_args,
                    momentum1,
                    momentum2,
                    iter_int,
                ),
            )
        if self.optimizer == OptimType.PARTIAL_ROWWISE_LAMB:
            return self._report_io_size_count(
                "fwd_output",
                invokers.lookup_partial_rowwise_lamb.invoke(
                    common_args,
                    self.optimizer_args,
                    momentum1,
                    momentum2,
                    iter_int,
                ),
            )

        prev_iter = invokers.lookup_args.Momentum(
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            dev=self.prev_iter_dev,
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            host=self.prev_iter_host,
            # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            uvm=self.prev_iter_uvm,
            # pyre-fixme[6]: For 4th argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            offsets=self.prev_iter_offsets,
            # pyre-fixme[6]: For 5th argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            placements=self.prev_iter_placements,
        )

        if self.optimizer == OptimType.EMAINPLACE_ROWWISE_ADAGRAD:
            with torch.no_grad():
                if self.training:
                    self.ema_inplace(self._emainplace_mode)
            return self._report_io_size_count(
                "fwd_output",
                invokers.lookup_rowwise_adagrad.invoke(
                    common_args,
                    self.optimizer_args,
                    momentum1,
                ),
            )

        if self.optimizer == OptimType.ENSEMBLE_ROWWISE_ADAGRAD:
            assert self._feature_is_enabled(
                FeatureGateName.TBE_ENSEMBLE_ROWWISE_ADAGRAD
            ), "ENSEMBLE_ROWWISE_ADAGRAD is an inactive or deprecated feature!"
            with torch.no_grad():
                if self.training:
                    self.ensemble_and_swap(self._ensemble_mode)
            return self._report_io_size_count(
                "fwd_output",
                invokers.lookup_rowwise_adagrad.invoke(
                    common_args,
                    self.optimizer_args,
                    momentum1,
                ),
            )

        if self._used_rowwise_adagrad_with_counter:
            if (
                self._max_counter_update_freq > 0
                and iter_int % self._max_counter_update_freq == 0
            ):
                row_counter_dev = self.row_counter_dev.detach()
                if row_counter_dev.numel() > 0:
                    self.max_counter[0] = torch.max(row_counter_dev).cpu().item() + 1
                else:
                    self.max_counter[0] = 1

        if self.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
            if self._used_rowwise_adagrad_with_counter:
                return self._report_io_size_count(
                    "fwd_output",
                    invokers.lookup_rowwise_adagrad_with_counter.invoke(
                        common_args,
                        self.optimizer_args,
                        momentum1,
                        prev_iter,
                        row_counter,
                        iter_int,
                        self.max_counter.item(),
                    ),
                )
            elif self._used_rowwise_adagrad_with_global_weight_decay:
                apply_global_weight_decay = (
                    iter_int >= self.gwd_start_iter and self.training
                )
                return self._report_io_size_count(
                    "fwd_output",
                    invokers.lookup_rowwise_adagrad.invoke(
                        common_args,
                        self.optimizer_args,
                        momentum1,
                        iter=iter_int,
                        apply_global_weight_decay=apply_global_weight_decay,
                        # pyre-fixme[6]: For 6th argument expected
                        #  `Optional[Tensor]` but got `Union[Module, Tensor]`.
                        prev_iter_dev=self.prev_iter_dev,
                        gwd_lower_bound=self.gwd_lower_bound,
                    ),
                )
            else:
                return self._report_io_size_count(
                    "fwd_output",
                    invokers.lookup_rowwise_adagrad.invoke(
                        common_args,
                        self.optimizer_args,
                        momentum1,
                    ),
                )

        raise ValueError(f"Invalid OptimType: {self.optimizer}")

    def ema_inplace(self, emainplace_mode: Dict[str, float]) -> None:
        """
        Perform ema operations on the full sparse embedding tables.
        We organize the sparse table, in the following way.

        Emb_table:
         -------------------------------------------------
         -                        --                     -
         -        Fast part       --      Slow part      -
         -    (RL) main part      --      target part    -
         -                        --                     -
         -------------------------------------------------

         In every "step_ema" step, we perform
            slow_part += coef_ema * (fast_part - slow_part)
        """
        iter_int = int(self.iter_cpu.item())
        if iter_int % int(emainplace_mode["step_ema"]) == 0 and iter_int >= int(
            emainplace_mode["step_start"]
        ):
            weights = self.split_embedding_weights()
            for table_i, (_, dim, _, _) in enumerate(self.embedding_specs):
                assert (
                    dim & 1 == 0
                ), f"table dimension {dim} is odd, not supported for ema_inplace_rowwise_adagrad"  # make sure that the dimension is even
                weights[table_i][:, dim // 2 :].data.lerp_(
                    weights[table_i][:, : dim // 2].data,
                    emainplace_mode["step_ema_coef"],
                )

    def ensemble_and_swap(self, ensemble_mode: Dict[str, float]) -> None:
        """
        Perform ensemble and swap operations on the full sparse embedding tables.

        Returns:
            Sparse embedding weights and optimizer states will be updated in-place.
        """
        iter_int = int(self.iter_cpu.item())
        should_ema = iter_int % int(ensemble_mode["step_ema"]) == 0
        should_swap = iter_int % int(ensemble_mode["step_swap"]) == 0
        if should_ema or should_swap:
            weights = self.split_embedding_weights()
            states = self.split_optimizer_states()
            coef_ema = (
                0.0
                if iter_int <= int(ensemble_mode["step_start"])
                else ensemble_mode["step_ema_coef"]
            )
            for i in range(len(self.embedding_specs)):
                # 0) copying weights from gpu to cpu
                weights_cpu = weights[i].to(
                    dtype=states[i][1].dtype, device=states[i][1].device
                )
                # 1) ema step
                if should_ema:
                    states[i][1].lerp_(weights_cpu, 1.0 - coef_ema)
                # 2) swap step
                if should_swap:
                    weights[i].copy_(states[i][1], non_blocking=True)
                # 3) post-processing step
                if should_ema:
                    if int(ensemble_mode["step_mode"]) == 0:  # embedding scaling
                        states[i][1].mul_(0.0)
                #  elif int(ensemble_mode["step_mode"]) == 2:  pure ema

    def reset_uvm_cache_stats(self) -> None:
        assert (
            self.gather_uvm_cache_stats
        ), "gather_uvm_cache_stats should be set to true to access uvm cache stats."
        self.uvm_cache_stats.zero_()
        self.local_uvm_cache_stats.zero_()

    def get_uvm_cache_stats(self, use_local_cache: bool = False) -> Tensor:
        assert (
            self.gather_uvm_cache_stats
        ), "gather_uvm_cache_stats should be set to true to access uvm cache stats."
        return self.local_uvm_cache_stats if use_local_cache else self.uvm_cache_stats

    def _get_uvm_cache_print_state(self, use_local_cache: bool = False) -> List[float]:
        snapshot = self.get_uvm_cache_stats(use_local_cache)
        if use_local_cache:
            return snapshot.tolist()

        # Stats are accumulated over multiple steps.  Compute delta, and update state.
        delta = snapshot - self.last_uvm_cache_print_state
        self.last_uvm_cache_print_state = snapshot.clone()
        return delta.tolist()

    @torch.jit.ignore
    def print_uvm_cache_stats(self, use_local_cache: bool = False) -> None:
        # TODO: Create a separate reporter class to unify the stdlog reporting
        uvm_cache_stats: List[float] = self._get_uvm_cache_print_state(use_local_cache)
        N = max(1, uvm_cache_stats[0])
        m = {
            "N_called": uvm_cache_stats[UVMCacheStatsIndex.num_calls],
            "requested_indices": uvm_cache_stats[
                UVMCacheStatsIndex.num_requested_indices
            ]
            / N,
            "unique_indices": uvm_cache_stats[UVMCacheStatsIndex.num_unique_indices]
            / N,
            "unique_misses": uvm_cache_stats[UVMCacheStatsIndex.num_unique_misses] / N,
            "conflict_unique_misses": uvm_cache_stats[
                UVMCacheStatsIndex.num_conflict_unique_misses
            ]
            / N,
            "conflict_misses": uvm_cache_stats[UVMCacheStatsIndex.num_conflict_misses]
            / N,
        }
        if uvm_cache_stats[1]:
            m.update(
                {
                    "unique indices / requested indices": uvm_cache_stats[
                        UVMCacheStatsIndex.num_unique_indices
                    ]
                    / uvm_cache_stats[UVMCacheStatsIndex.num_requested_indices],
                    "unique misses / requested indices": uvm_cache_stats[
                        UVMCacheStatsIndex.num_unique_misses
                    ]
                    / uvm_cache_stats[UVMCacheStatsIndex.num_requested_indices],
                }
            )
        self.log(f"uvm_cache_stats={m}")

    @torch.jit.ignore
    def _report_uvm_cache_stats(self) -> None:
        if self.stats_reporter is None:
            return
        stats_reporter: TBEStatsReporter = self.stats_reporter
        passed_steps = self.step - self.last_reported_step
        if passed_steps == 0:
            return
        if not stats_reporter.should_report(self.step):
            return

        uvm_cache_stats: List[float] = self.get_uvm_cache_stats(
            use_local_cache=False
        ).tolist()
        self.last_reported_step = self.step

        if len(self.last_reported_uvm_stats) == 0:
            self.last_reported_uvm_stats = [0.0] * len(uvm_cache_stats)
        uvm_cache_stats_delta: List[float] = [0.0] * len(uvm_cache_stats)
        for i in range(len(uvm_cache_stats)):
            uvm_cache_stats_delta[i] = (
                uvm_cache_stats[i] - self.last_reported_uvm_stats[i]
            )
        self.last_reported_uvm_stats = uvm_cache_stats

        # pyre-fixme[29]: `Union[(self: TensorBase) -> int, Module, Tensor]` is not
        #  a function.
        element_size = self.lxu_cache_weights.element_size()
        for stat_index in UVMCacheStatsIndex:
            stats_reporter.report_data_amount(
                iteration_step=self.step,
                event_name=f"tbe.prefetch.cache_stats_by_data_size.{stat_index.name.lower()}",
                data_bytes=int(
                    uvm_cache_stats_delta[stat_index.value]
                    * element_size
                    * self.max_D_cache
                    / passed_steps
                ),
                embedding_id=self.logging_table_name,
                tbe_id=self.uuid,
            )

    def prefetch(
        self,
        indices: Tensor,
        offsets: Tensor,
        forward_stream: Optional[torch.cuda.Stream] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
    ) -> None:
        if self.prefetch_stream is None and forward_stream is not None:
            self.prefetch_stream = torch.cuda.current_stream()
            assert (
                self.prefetch_stream != forward_stream
            ), "prefetch_stream and forward_stream should not be the same stream"

        indices, offsets, _, vbe_metadata = self.prepare_inputs(
            indices,
            offsets,
            per_sample_weights=None,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            force_cast_input_types=False,
            prefetch_pipeline=self.prefetch_pipeline,
        )

        with self._recording_to_timer(
            self.prefetch_duration_timer,
            context=self.step,
            stream=torch.cuda.current_stream(),
        ):
            self._prefetch(
                indices,
                offsets,
                vbe_metadata,
                multipass_prefetch_config=self.multipass_prefetch_config,
            )

        if forward_stream is not None:
            self._prefetch_tensors_record_stream(forward_stream)

    def _prefetch(
        self,
        indices: Tensor,
        offsets: Tensor,
        vbe_metadata: Optional[invokers.lookup_args.VBEMetadata] = None,
        multipass_prefetch_config: Optional[MultiPassPrefetchConfig] = None,
    ) -> None:
        if not is_torchdynamo_compiling():
            # Mutations of nn.Module attr forces dynamo restart of Analysis which increases compilation time
            self.timestep += 1
            self.timesteps_prefetched.append(self.timestep)

        # pyre-fixme[29]: `Union[(self: TensorBase) -> int, Module, Tensor]` is not
        #  a function.
        if not self.lxu_cache_weights.numel():
            return

        # Clear the local_uvm_cache_stats before the prefetch instead of after
        # the prefetch step, since it will be used in the CommonArgs in the
        # forward step
        if self.gather_uvm_cache_stats:
            self.local_uvm_cache_stats.zero_()
        self._report_io_size_count("prefetch_input", indices)

        final_lxu_cache_locations = torch.empty_like(indices, dtype=torch.int32)
        for (
            partial_indices,
            partial_lxu_cache_locations,
            base_offset,
        ) in self.get_prefetch_passes(
            multipass_prefetch_config, indices, final_lxu_cache_locations
        ):
            linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
                self.cache_hash_size_cumsum,
                partial_indices,
                offsets,
                vbe_metadata.B_offsets if vbe_metadata is not None else None,
                vbe_metadata.max_B if vbe_metadata is not None else -1,
                base_offset,
            )

            if (
                self.record_cache_metrics.record_cache_miss_counter
                or self.record_cache_metrics.record_tablewise_cache_miss
            ):
                lxu_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
                    linear_cache_indices,
                    self.lxu_cache_state,
                    self.total_cache_hash_size,
                    self.gather_uvm_cache_stats,
                    self.local_uvm_cache_stats,
                )
                if self.record_cache_metrics.record_cache_miss_counter:
                    self._update_cache_miss_counter(
                        lxu_cache_locations, linear_cache_indices
                    )
                if self.record_cache_metrics.record_tablewise_cache_miss:
                    self._update_tablewise_cache_miss(
                        lxu_cache_locations, linear_cache_indices, offsets
                    )

            if self.cache_algorithm == CacheAlgorithm.LRU:
                torch.ops.fbgemm.lru_cache_populate(
                    self.weights_uvm,
                    self.cache_hash_size_cumsum,
                    self.total_cache_hash_size,
                    self.cache_index_table_map,
                    self.weights_offsets,
                    self.D_offsets,
                    linear_cache_indices,
                    self.lxu_cache_state,
                    self.lxu_cache_weights,
                    self.timestep,
                    self.lxu_state,
                    self.stochastic_rounding,
                    self.gather_uvm_cache_stats,
                    self.local_uvm_cache_stats,
                    self.lock_cache_line,
                    self.lxu_cache_locking_counter,
                )
            elif self.cache_algorithm == CacheAlgorithm.LFU:
                torch.ops.fbgemm.lfu_cache_populate(
                    self.weights_uvm,
                    self.cache_hash_size_cumsum,
                    self.total_cache_hash_size,
                    self.cache_index_table_map,
                    self.weights_offsets,
                    self.D_offsets,
                    linear_cache_indices,
                    self.lxu_cache_state,
                    self.lxu_cache_weights,
                    self.lxu_state,
                    self.stochastic_rounding,
                )

            torch.ops.fbgemm.lxu_cache_lookup(
                linear_cache_indices,
                self.lxu_cache_state,
                self.total_cache_hash_size,
                self.gather_uvm_cache_stats,
                self.local_uvm_cache_stats,
                lxu_cache_locations_output=partial_lxu_cache_locations,
            )

        assert (
            len(self.lxu_cache_locations_list) < self.max_prefetch_depth
        ), f"self.lxu_cache_locations_list has grown to size: {len(self.lxu_cache_locations_list)}, this exceeds the maximum: {self.max_prefetch_depth}. This probably indicates an error in logic where prefetch() is being called more frequently than forward()"
        self.lxu_cache_locations_list.append(final_lxu_cache_locations)

        if self.gather_uvm_cache_stats:
            # Accumulate local_uvm_cache_stats (int32) into uvm_cache_stats (int64).
            # We may want to do this accumulation atomically, but as it's only
            # for monitoring, slightly inaccurate result may be acceptable.
            self.uvm_cache_stats = torch.add(
                self.uvm_cache_stats, self.local_uvm_cache_stats
            )
            self._report_uvm_cache_stats()
            if self.should_log():
                self.print_uvm_cache_stats(use_local_cache=False)

    def should_log(self) -> bool:
        """Determines if we should log for this step, using exponentially decreasing frequency.

        Logs for steps: 100 200 ... 1,000 2,000 ... 10,000 20,000 ... 100,000 200,000 ...
        """
        s = self.step + 1  # step starts at 0
        return s >= 100 and s % (10 ** int(math.log10(s))) == 0

    def _prefetch_tensors_record_stream(
        self, forward_stream: torch.cuda.Stream
    ) -> None:
        # Record the tensors created by prefetch stream and consumed by forward/backward
        # to the forward stream. In PyTorch, each backward CUDA op runs on the same
        # stream that was used for its corresponding forward op.

        for t in self.lxu_cache_locations_list:
            t.record_stream(forward_stream)

    def _update_cache_miss_counter(
        self,
        lxu_cache_locations: Tensor,
        linear_cache_indices: Tensor,
    ) -> None:
        CACHE_MISS = -1
        CACHE_HIT = -2

        cache_missed_locations = torch.where(
            lxu_cache_locations == CACHE_MISS, linear_cache_indices, CACHE_HIT
        )
        unique_ids_list = torch.unique(cache_missed_locations)
        unique_ids_count_list = torch.where(unique_ids_list == CACHE_HIT, 0, 1)

        miss_count = torch.sum(unique_ids_count_list)

        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.cache_miss_counter[0] += (miss_count > 0).to(torch.int64)

        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.cache_miss_counter[1] += miss_count

    def _update_tablewise_cache_miss(
        self,
        lxu_cache_locations: Tensor,
        linear_cache_indices: Tensor,
        offsets: Tensor,
    ) -> None:
        CACHE_MISS = -1
        CACHE_HIT = -2

        # pyre-fixme[6]: For 1st argument expected
        #  `pyre_extensions.PyreReadOnly[Sized]` but got `Union[Module, Tensor]`.
        num_tables = len(self.cache_hash_size_cumsum) - 1
        num_offsets_per_table = (len(offsets) - 1) // num_tables
        cache_missed_locations = torch.where(
            lxu_cache_locations == CACHE_MISS, linear_cache_indices, CACHE_HIT
        )

        for i in range(num_tables):
            start = offsets[i * num_offsets_per_table]
            end = offsets[(i + 1) * num_offsets_per_table]

            current_cache_missed_locations = cache_missed_locations[start:end]
            unique_ids_list = torch.unique(current_cache_missed_locations)
            unique_ids_count_list = torch.where(unique_ids_list == CACHE_HIT, 0, 1)

            miss_count = torch.sum(unique_ids_count_list)

            self.table_wise_cache_miss[i] += miss_count

    def init_embedding_weights_uniform(self, min_val: float, max_val: float) -> None:
        splits = self.split_embedding_weights()
        if self.weights_precision == SparseType.INT8:
            # TODO: add in-place FloatToFused8BitRowwiseQuantized conversion
            for emb in splits:
                assert (
                    len(emb.shape) == 2
                ), "Int8 embedding only supported for 2D weight tensors."
                shape = [emb.shape[0], emb.shape[1] - self.int8_emb_row_dim_offset]
                tmp_emb = torch.zeros(shape, device=self.current_device)
                tmp_emb.uniform_(min_val, max_val)
                tmp_emb_i8 = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(tmp_emb)
                emb.data.copy_(tmp_emb_i8)
        else:
            for param in splits:
                param.uniform_(min_val, max_val)

    @torch.jit.ignore
    def split_embedding_weights(self) -> List[Tensor]:
        """
        Returns a list of embedding weights (view), split by table

        Returns:
            A list of weights. Length = the number of tables
        """
        splits = []
        for t, (rows, dim, _, _) in enumerate(self.embedding_specs):
            if self.weights_precision == SparseType.INT8:
                dim += self.int8_emb_row_dim_offset
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[An...
            placement = self.weights_physical_placements[t]
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[An...
            offset = self.weights_physical_offsets[t]
            if placement == EmbeddingLocation.DEVICE.value:
                weights = self.weights_dev
            elif placement == EmbeddingLocation.HOST.value:
                weights = self.weights_host
            else:
                weights = self.weights_uvm
            # pyre-fixme[29]: `Union[(self: TensorBase) -> int, Module, Tensor]` is
            #  not a function.
            if weights.dim() == 2:
                weights = weights.flatten()
            splits.append(
                weights.detach()[offset : offset + rows * dim].view(rows, dim)
            )
        return splits

    @torch.jit.ignore
    def get_optimizer_buffer(self, state: str) -> torch.Tensor:
        if self.optimizer == OptimType.NONE:
            raise NotImplementedError(
                f"Getting optimizer buffer is not supported for {self.optimizer}"
            )
        for name, buffer in self.named_buffers():
            if name == state:
                return buffer
        raise ValueError(f"Optimizer buffer {state} not found")

    @torch.jit.export
    def get_optimizer_state(self) -> List[Dict[str, torch.Tensor]]:
        r"""
        Get the optimizer state dict that matches the OSS Pytorch optims
        TODO: populate the supported list of optimizers
        """
        split_optimizer_states = self.split_optimizer_states()
        if (
            self.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD
            or self.optimizer == OptimType.EXACT_ADAGRAD
            or self.optimizer == OptimType.EMAINPLACE_ROWWISE_ADAGRAD
        ):
            list_of_state_dict = [
                (
                    (
                        {
                            "sum": states[0],
                            "prev_iter": states[1],
                            "row_counter": states[2],
                            "iter": self.iter,
                        }
                        if self.optimizer_args.regularization_mode
                        == WeightDecayMode.COUNTER.value
                        and self.optimizer_args.weight_decay_mode
                        == CounterWeightDecayMode.ADAGRADW.value
                        else {
                            "sum": states[0],
                            "prev_iter": states[1],
                            "row_counter": states[2],
                        }
                    )
                    if self._used_rowwise_adagrad_with_counter
                    else (
                        {
                            "sum": states[0],
                            "prev_iter": states[1],
                            "iter": self.iter,
                        }
                        if self._used_rowwise_adagrad_with_global_weight_decay
                        else {"sum": states[0]}
                    )
                )
                for states in split_optimizer_states
            ]
        elif self.optimizer == OptimType.SGD or self.optimizer == OptimType.EXACT_SGD:
            list_of_state_dict = [
                {"momentum_buffer": states[0]} for states in split_optimizer_states
            ]
        elif self.optimizer == OptimType.ADAM and self.use_rowwise_bias_correction:
            list_of_state_dict = [
                {
                    "exp_avg": states[0],
                    "exp_avg_sq": states[1],
                    "row_counter": states[2],
                }
                for states in split_optimizer_states
            ]
        elif (
            self.optimizer == OptimType.ADAM
            or self.optimizer == OptimType.PARTIAL_ROWWISE_ADAM
            or self.optimizer == OptimType.LAMB
            or self.optimizer == OptimType.PARTIAL_ROWWISE_LAMB
        ):
            list_of_state_dict = [
                {"exp_avg": states[0], "exp_avg_sq": states[1]}
                for states in split_optimizer_states
            ]
        elif self.optimizer == OptimType.ENSEMBLE_ROWWISE_ADAGRAD:
            list_of_state_dict = [
                {
                    "sum": states[0],
                    "sparse_ema": states[1],
                }
                for states in split_optimizer_states
            ]
        else:
            raise NotImplementedError(
                f"Getting optimizer state {self.optimizer} is not implmeneted"
            )

        return list_of_state_dict

    @torch.jit.ignore
    def split_optimizer_states(
        self,
    ) -> List[List[torch.Tensor]]:
        """
        Returns a list of optimizer states (view), split by table

        Returns:
            A list of list of states. Shape = (the number of tables, the number
            of states).

            The following shows the list of states (in the returned order) for
            each optimizer:

            (1) `ADAM`: `momentum1`, `momentum2`

            (2) `EXACT_ADAGRAD`: `momentum1`

            (3) `EXACT_ROWWISE_ADAGRAD`: `momentum1` (rowwise), `prev_iter`
                (rowwise; only when using `WeightDecayMode` = `COUNTER` or
                `COWCLIP` or `global_weight_decay` is not None), `row_counter`
                (rowwise; only when using `WeightDecayMode` = `COUNTER` or
                `COWCLIP`)

            (4) `EXACT_SGD`: no states

            (5) `LAMB`: `momentum1`, `momentum2`

            (6) `LARS_SGD`: `momentum1`

            (7) `PARTIAL_ROWWISE_ADAM`: `momentum1`, `momentum2` (rowwise)

            (8) `PARTIAL_ROWWISE_LAMB`: `momentum1`, `momentum2` (rowwise)

            (9) `ENSEMBLE_ROWWISE_ADAGRAD`: `momentum1` (rowwise), `momentum2`

            (10) `NONE`: no states (throwing an error)

        """
        if self.optimizer == OptimType.NONE:
            raise NotImplementedError(
                f"Getting optimizer states is not supported for {self.optimizer}"
            )

        def get_optimizer_states(
            state_dev: Tensor,
            state_host: Tensor,
            state_uvm: Tensor,
            state_offsets: Tensor,
            state_placements: Tensor,
            rowwise: bool,
        ) -> List[torch.Tensor]:
            splits = []
            for t, (rows, dim, _, _) in enumerate(self.embedding_specs):
                offset = state_offsets[t]
                placement = state_placements[t]
                if placement == EmbeddingLocation.DEVICE:
                    state = state_dev
                elif placement == EmbeddingLocation.HOST:
                    state = state_host
                else:
                    state = state_uvm
                if not rowwise:
                    splits.append(
                        state.detach()[offset : offset + rows * dim].view(rows, dim)
                    )
                else:
                    splits.append(state.detach()[offset : offset + rows].view(rows))
            return splits

        states: List[List[torch.Tensor]] = []
        if self.optimizer not in (OptimType.EXACT_SGD,):
            states.append(
                get_optimizer_states(
                    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.momentum1_dev,
                    # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.momentum1_host,
                    # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.momentum1_uvm,
                    # pyre-fixme[6]: For 4th argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.momentum1_physical_offsets,
                    # pyre-fixme[6]: For 5th argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.momentum1_physical_placements,
                    rowwise=self.optimizer
                    in [
                        OptimType.EXACT_ROWWISE_ADAGRAD,
                        OptimType.ENSEMBLE_ROWWISE_ADAGRAD,
                        OptimType.EMAINPLACE_ROWWISE_ADAGRAD,
                    ],
                )
            )
        if self.optimizer in (
            OptimType.ADAM,
            OptimType.PARTIAL_ROWWISE_ADAM,
            OptimType.LAMB,
            OptimType.PARTIAL_ROWWISE_LAMB,
            OptimType.ENSEMBLE_ROWWISE_ADAGRAD,
        ):
            states.append(
                get_optimizer_states(
                    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.momentum2_dev,
                    # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.momentum2_host,
                    # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.momentum2_uvm,
                    # pyre-fixme[6]: For 4th argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.momentum2_physical_offsets,
                    # pyre-fixme[6]: For 5th argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.momentum2_physical_placements,
                    rowwise=self.optimizer
                    in (OptimType.PARTIAL_ROWWISE_ADAM, OptimType.PARTIAL_ROWWISE_LAMB),
                )
            )
        if (
            self._used_rowwise_adagrad_with_counter
            or self._used_rowwise_adagrad_with_global_weight_decay
        ):
            states.append(
                get_optimizer_states(
                    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.prev_iter_dev,
                    # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.prev_iter_host,
                    # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.prev_iter_uvm,
                    # pyre-fixme[6]: For 4th argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.prev_iter_physical_offsets,
                    # pyre-fixme[6]: For 5th argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.prev_iter_physical_placements,
                    rowwise=True,
                )
            )
        if self._used_rowwise_adagrad_with_counter or (
            self.optimizer == OptimType.ADAM and self.use_rowwise_bias_correction
        ):
            states.append(
                get_optimizer_states(
                    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.row_counter_dev,
                    # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.row_counter_host,
                    # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.row_counter_uvm,
                    # pyre-fixme[6]: For 4th argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.row_counter_physical_offsets,
                    # pyre-fixme[6]: For 5th argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    self.row_counter_physical_placements,
                    rowwise=True,
                )
            )
        return_states = [list(s) for s in zip(*states)]
        return return_states

    @torch.jit.export
    def set_learning_rate(self, lr: float) -> None:
        """
        Sets the learning rate.

        Args:
            lr (float): The learning rate value to set to
        """
        if self.optimizer == OptimType.NONE:
            raise NotImplementedError(
                f"Setting learning rate is not supported for {self.optimizer}"
            )
        self._set_learning_rate(lr)

    def get_learning_rate(self) -> float:
        """
        Get and return the learning rate.
        """
        return self.learning_rate_tensor.item()

    @torch.jit.ignore
    def update_hyper_parameters(self, params_dict: Dict[str, float]) -> None:
        """
        Sets hyper-parameters from external control flow.

        Args:
            params_dict (Dict[str, float]): The dict that contains the
                hyper-parameter names and their values
        """
        if self.optimizer == OptimType.NONE:
            raise NotImplementedError(
                f"Setting learning rate is not supported for {self.optimizer}"
            )
        for parameter_name, value in params_dict.items():
            if parameter_name == "lr":
                self._set_learning_rate(value)
            elif parameter_name == "eps":
                self.optimizer_args = self.optimizer_args._replace(eps=value)
            elif parameter_name == "beta1":
                self.optimizer_args = self.optimizer_args._replace(beta1=value)
            elif parameter_name == "beta2":
                self.optimizer_args = self.optimizer_args._replace(beta2=value)
            elif parameter_name == "weight_decay":
                self.optimizer_args = self.optimizer_args._replace(weight_decay=value)
            elif parameter_name == "lower_bound":
                self.gwd_lower_bound = value
            else:
                raise NotImplementedError(
                    f"Setting hyper-parameter {parameter_name} is not supported"
                )

    @torch.jit.ignore
    def _set_learning_rate(self, lr: float) -> float:
        """
        Helper function to script `set_learning_rate`.
        Note that returning None does not work.
        """
        self.learning_rate_tensor.fill_(lr)
        return 0.0

    @torch.jit.ignore
    def set_optimizer_step(self, step: int) -> None:
        """
        Sets the optimizer step.

        Args:
            step (int): The step value to set to
        """
        self.log(f"set_optimizer_step from {self.iter[0]=} to {step=}")
        if self.optimizer == OptimType.NONE:
            raise NotImplementedError(
                f"Setting optimizer step is not supported for {self.optimizer}"
            )
        self.iter[0] = step

    @torch.jit.export
    def flush(self) -> None:
        # pyre-fixme[29]: `Union[(self: TensorBase) -> int, Module, Tensor]` is not
        #  a function.
        if not self.lxu_cache_weights.numel():
            return
        torch.ops.fbgemm.lxu_cache_flush(
            self.weights_uvm,
            self.cache_hash_size_cumsum,
            self.cache_index_table_map,
            self.weights_offsets,
            self.D_offsets,
            self.total_D,
            self.lxu_cache_state,
            self.lxu_cache_weights,
            self.stochastic_rounding,
        )

    def _apply_split(
        self,
        split: SplitState,
        prefix: str,
        dtype: Type[torch.dtype],
        enforce_hbm: bool = False,
        make_dev_param: bool = False,
        dev_reshape: Optional[Tuple[int, ...]] = None,
        uvm_host_mapped: bool = False,
    ) -> None:
        apply_split_helper(
            self.register_buffer,
            functools.partial(setattr, self),
            self.current_device,
            self.use_cpu,
            self.feature_table_map,
            split,
            prefix,
            dtype,
            enforce_hbm,
            make_dev_param,
            dev_reshape,
            self._uvm_tensors_log,
            uvm_host_mapped=uvm_host_mapped,
        )

    def _apply_cache_state(
        self,
        cache_state: CacheState,
        cache_algorithm: CacheAlgorithm,
        cache_load_factor: float,
        cache_sets: int,
        cache_reserved_memory: float,
        cache_precision: SparseType,
    ) -> None:
        self.cache_algorithm = cache_algorithm
        self.timestep = 1
        self.timesteps_prefetched = []

        self.max_prefetch_depth = MAX_PREFETCH_DEPTH
        self.lxu_cache_locations_list = []
        self.lxu_cache_locations_empty = torch.empty(
            0, device=self.current_device, dtype=torch.int32
        ).fill_(-1)
        self.lxu_cache_locations = self.lxu_cache_locations_empty
        self._indices = self.lxu_cache_locations_empty
        self._offsets = self.lxu_cache_locations_empty
        self._vbe_B_offsets = self.lxu_cache_locations_empty
        self._vbe_max_B = -1
        self.prefetch_stream: Optional[torch.cuda.Stream] = None

        self._init_uvm_cache_stats()

        if cache_precision == SparseType.FP32:
            dtype = torch.float32
        elif cache_precision == SparseType.FP16:
            dtype = torch.float16
        else:
            dtype = torch.float32  # not relevant, but setting it to keep linter happy
            if not self.use_cpu > 0:
                raise AssertionError(
                    f"cache_precision {cache_precision} not supported!"
                )

        # NOTE: no cache for CPU mode!
        if cache_state.total_cache_hash_size == 0 or self.use_cpu:
            self.register_buffer(
                "lxu_cache_weights",
                torch.zeros(0, 0, device=self.current_device, dtype=dtype),
            )
            # NOTE: make TorchScript work!
            self.register_buffer(
                "cache_hash_size_cumsum",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "total_cache_hash_size",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "cache_index_table_map",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "lxu_cache_state",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "lxu_state",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "cache_miss_counter",
                torch.tensor([0, 0], dtype=torch.int64),
                persistent=False,
            )
            self._init_uvm_cache_counter(cache_sets, persistent=False)
            return

        assert cache_load_factor > 0
        element_size = 2 if dtype == torch.float16 else 4
        if cache_sets <= 0:
            total_memory = torch.cuda.get_device_properties(
                self.current_device
            ).total_memory
            free_memory = (
                total_memory
                - torch.cuda.memory_reserved(self.current_device)
                - int(cache_reserved_memory)
            )
            assert free_memory > 0
            cache_sets = (
                int(cache_state.total_cache_hash_size * cache_load_factor)
                + DEFAULT_ASSOC
                - 1
            ) // DEFAULT_ASSOC
            cache_sets = 1 if cache_sets == 0 else cache_sets
            cache_size = cache_sets * DEFAULT_ASSOC * element_size * self.max_D_cache
            if cache_size > free_memory:
                cache_sets = (
                    int(1.0 * free_memory / self.max_D_cache / element_size)
                    + DEFAULT_ASSOC
                    - 1
                ) // DEFAULT_ASSOC
        cache_load_factor = (
            1.0 * cache_sets * DEFAULT_ASSOC / int(cache_state.total_cache_hash_size)
        )
        assert cache_sets > 0
        if cache_algorithm == CacheAlgorithm.LFU:
            assert cache_sets < 2**24 - 1
        cache_size = cache_sets * DEFAULT_ASSOC * element_size * self.max_D_cache
        self.log(
            f"Using on-device cache with admission algorithm "
            f"{cache_algorithm}, {cache_sets} sets, "
            f"load_factor: {cache_load_factor : .3f}, "
            f"cache_size: {cache_size / 1024.0 / 1024.0 / 1024.0 : .2f}GB, "
            f"cache_precision: {dtype}, "
            f"weights_precision: {self.weights_precision}"
        )

        self.total_cache_hash_size = cache_state.total_cache_hash_size
        # 8x of # tables, trivial size
        self.register_buffer(
            "cache_hash_size_cumsum",
            torch.tensor(
                cache_state.cache_hash_size_cumsum,
                device=self.current_device,
                dtype=torch.int64,
            ),
        )
        # 4x total embedding hash size with uvm cache
        self.register_buffer(
            "cache_index_table_map",
            torch.tensor(
                cache_state.cache_index_table_map,
                device=self.current_device,
                dtype=torch.int32,
            ),
        )
        # 8x of total cache slots (embedding hash size * clf)
        self.register_buffer(
            "lxu_cache_state",
            torch.zeros(
                cache_sets, DEFAULT_ASSOC, device=self.current_device, dtype=torch.int64
            ).fill_(-1),
        )
        # Cache itself, not auxiliary size
        self.register_buffer(
            "lxu_cache_weights",
            torch.zeros(
                cache_sets * DEFAULT_ASSOC,
                self.max_D_cache,
                device=self.current_device,
                dtype=dtype,
            ),
        )
        # LRU: 8x of total cache slots (embedding hash size * clf)
        # LFU: 8x of total embedding hash size with uvm cache
        self.register_buffer(
            "lxu_state",
            torch.zeros(
                size=(
                    (self.total_cache_hash_size + 1,)
                    if cache_algorithm == CacheAlgorithm.LFU
                    else (cache_sets, DEFAULT_ASSOC)
                ),
                device=self.current_device,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "cache_miss_counter",
            torch.tensor([0, 0], device=self.current_device, dtype=torch.int64),
        )
        self._init_uvm_cache_counter(cache_sets, persistent=True)
        if self.prefetch_pipeline:
            # using the placeholder_autograd_tensor to make sure
            # the hook is executed after the backward pass
            # not using register_module_full_backward_hook
            # due to https://github.com/pytorch/pytorch/issues/100528
            self.placeholder_autograd_tensor.register_hook(
                self._sync_stream_post_backward
            )
            self.register_full_backward_pre_hook(
                self._update_cache_counter_and_locations
            )

        if cache_algorithm not in (CacheAlgorithm.LFU, CacheAlgorithm.LRU):
            raise ValueError(
                f"cache_algorithm must be {CacheAlgorithm.LRU} "
                f"or {CacheAlgorithm.LFU}"
            )

    # pyre-ignore
    def _recording_to_timer(
        self, timer: Optional[AsyncSeriesTimer], **kwargs: Any
    ) -> Any:
        if self.stats_reporter is not None and self.stats_reporter.should_report(
            self.step
        ):
            assert (
                timer
            ), "We shouldn't be here, async timer must have been initiated if reporter is present."
            return timer.recording(**kwargs)
        # No-Op context manager
        return contextlib.nullcontext()

    def _sync_stream_post_backward(
        self,
        grad: Tensor,
    ) -> None:
        """
        backward hook function when prefetch_pipeline is enabled.

        With the pipeline, prefetch(batch_{i+2}) may overlap with backward(batch_{i}).
        There is race condition that backward(batch_i) writes to UVM memory and
        at the same time prefetch(batch_{i+2}) loads UVM memory to cache. This stream sync forces
        backward(batch_i) to finish before prefetch(batch_{i+2}).
        """
        if self.prefetch_stream is not None:
            self.prefetch_stream.wait_stream(torch.cuda.current_stream())

    def _update_cache_counter_and_locations(
        self,
        module: nn.Module,
        grad_input: Union[Tuple[Tensor, ...], Tensor],
    ) -> None:
        """
        Backward prehook function when prefetch_pipeline is enabled.

        This function does 3 things:
        1. backward stream waits for prefetch stream to finish.
        Otherwise the prefetch(batch_{i+1}) might overlap with backward(batch_i).
        If an idx is not in cache in batch_i, but it is being inserted in batch_{i+1},
        there is race condition that backward(batch_i) writes to UVM memory and
        at the same time prefetch(batch_{i+1}) loads UVM memory to cache.

        2. decrement the lxu_cache_locking_counter to indicate the current batch is finished.
        The lxu_cache_locking_counter is updated in both prefetch and TBE backward.
        As there is no overlap between prefetch and backward, we can decrement either before or
        after backward. It's better to decrement before lxu_cache_locations gets updated.

        3. update lxu_cache_locations to address the cache inconsistency issue.
        In the case that the same index is not inserted into cache in batch_i,
        but it is inserted in batch_{i+1}, the cache can be invalid in
        the sense that the cached weight for this index does not have the
        backward update of batch_i.

        Example of the issue is as follows:
        idx is in batch_i, batch_{i+1}
        prefetch(batch_i)
          - failed to insert idx into cache, cache_locations_batch_i of idx is -1 (cache miss)
        forward(batch_i)
        prefetch(batch_{i+1})
          - insert idx into cache, cache is loaded from host memory
        backward(batch_i)
          - cache_locations_batch_i of idx is -1, the host memory is updated
        forward(batch_{i+1})
          - OUTPUT IS WRONG. the weight for idx is fetched from cache, but the cache is outdated.

        The fix to this cache inconsistency is to update the cache_locations_batch_i before backward of batch_i,
        so that the cache gets updated correctly by the backward pass of TBE.
        """

        if self.prefetch_stream is not None:
            # need to wait for the prefetch of next batch,
            # so that cache states are valid
            with self._recording_to_timer(
                self.bwd_wait_prefetch_timer,
                context=self.step,
                stream=torch.cuda.current_stream(),
            ):
                torch.cuda.current_stream().wait_stream(self.prefetch_stream)

        torch.ops.fbgemm.lxu_cache_locking_counter_decrement(
            self.lxu_cache_locking_counter,
            self.lxu_cache_locations,
        )
        # Recompute linear_cache_indices
        linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
            self.cache_hash_size_cumsum,
            self._indices,
            self._offsets,
            self._vbe_B_offsets,
            self._vbe_max_B,
        )
        (
            linear_unique_indices,
            linear_unique_indices_length,
            _,
        ) = torch.ops.fbgemm.get_unique_indices(
            linear_cache_indices,
            self.total_cache_hash_size,
            compute_count=False,
        )
        torch.ops.fbgemm.lxu_cache_lookup(
            linear_unique_indices,
            self.lxu_cache_state,
            self.total_cache_hash_size,
            gather_cache_stats=False,  # not collecting cache stats
            num_uniq_cache_indices=linear_unique_indices_length,
            lxu_cache_locations_output=self.lxu_cache_locations,
        )

    def _init_uvm_cache_counter(self, cache_sets: int, persistent: bool) -> None:
        if self.prefetch_pipeline and persistent:
            self.register_buffer(
                "lxu_cache_locking_counter",
                torch.zeros(
                    cache_sets,
                    DEFAULT_ASSOC,
                    device=self.current_device,
                    dtype=torch.int32,
                ),
            )
        else:
            self.register_buffer(
                "lxu_cache_locking_counter",
                torch.zeros([0, 0], dtype=torch.int32, device=self.current_device),
                persistent=persistent,
            )

    def _init_uvm_cache_stats(self) -> None:
        if not self.gather_uvm_cache_stats:
            # If uvm_cache_stats is not enabled, register stub entries via buffer to state_dict for TorchScript to JIT properly.
            # Since we're not using these variables, we can choose minimize tensor size to keep state_dict size small.
            self.register_buffer(
                "uvm_cache_stats",
                torch.zeros(
                    1,
                    device=self.current_device,
                    dtype=torch.int64,
                ),
                persistent=False,
            )
            self.register_buffer(
                "local_uvm_cache_stats",
                torch.zeros(
                    1,
                    device=self.current_device,
                    dtype=torch.int32,
                ),
                persistent=False,
            )
        else:
            self.register_buffer(
                "uvm_cache_stats",
                torch.zeros(
                    size=(self.uvm_cache_stats_size,),
                    device=self.current_device,
                    dtype=torch.int64,
                ),
            )
            self.register_buffer(
                "local_uvm_cache_stats",
                torch.zeros(
                    size=(self.uvm_cache_stats_size,),
                    device=self.current_device,
                    dtype=torch.int32,
                ),
            )
            self.reset_uvm_cache_stats()
        self.last_uvm_cache_print_state = torch.zeros_like(self.uvm_cache_stats)

    def reset_cache_states(self) -> None:
        # pyre-fixme[29]: `Union[(self: TensorBase) -> int, Module, Tensor]` is not
        #  a function.
        if not self.lxu_cache_weights.numel():
            return
        self.lxu_cache_state.fill_(-1)
        self.lxu_state.fill_(0)
        self.timestep = 1

    def reset_embedding_weight_momentum(
        self,
        pruned_indices: Tensor,
        pruned_indices_offsets: Tensor,
        logical_table_ids: Tensor,
        buffer_ids: Tensor,
    ) -> None:
        if self.optimizer == OptimType.NONE:
            raise NotImplementedError(
                f"Resetting embedding weight momentum is not supported for {self.optimizer}"
            )
        total_cache_hash_size = 0
        if isinstance(self.total_cache_hash_size, Tensor):
            total_cache_hash_size = self.total_cache_hash_size.item()
        else:
            total_cache_hash_size = self.total_cache_hash_size

        rowwise = self.optimizer in [
            OptimType.EXACT_ROWWISE_ADAGRAD,
        ]
        if rowwise:
            torch.ops.fbgemm.reset_weight_momentum(
                dev_weights=self.weights_dev,
                uvm_weights=self.weights_uvm,
                lxu_cache_weights=self.lxu_cache_weights,
                weights_placements=self.weights_placements,
                weights_offsets=self.weights_offsets,
                momentum1_dev=self.momentum1_dev,
                momentum1_uvm=self.momentum1_uvm,
                momentum1_placements=self.momentum1_placements,
                momentum1_offsets=self.momentum1_offsets,
                D_offsets=self.D_offsets,
                pruned_indices=pruned_indices.to(device=self.current_device),
                pruned_indices_offsets=pruned_indices_offsets.to(
                    device=self.current_device
                ),
                logical_table_ids=logical_table_ids.to(device=self.current_device),
                buffer_ids=buffer_ids.to(device=self.current_device),
                cache_hash_size_cumsum=self.cache_hash_size_cumsum,
                lxu_cache_state=self.lxu_cache_state,
                total_cache_hash_size=total_cache_hash_size,
            )

    def prepare_inputs(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        force_cast_input_types: bool = True,
        prefetch_pipeline: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], invokers.lookup_args.VBEMetadata]:
        """
        Prepare TBE inputs as follows:

        (1) Create VBE metadata
        (2) Convert input types if `force_cast_input_types=True`
        (3) Run `bounds_check_indices` if `bounds_check_mode` is not
            BoundsCheckMode.NONE

        Args:
            indices (Tensor): Input indices
            offsets (Tensor): Input offsets
            per_sample_weights (Optional[Tensor]): Input per sample
                weights
            batch_size_per_feature_per_rank
                (Optional[List[List[int]]]): A 2D tensor of batch size
                for each rank and feature. Shape = (number of
                features, number of ranks)
            force_cast_input_types (bool): A flag to force convert
                input types if set to True

        Returns:
            A tuple of indices, offsets, per_sample_weights, and VBE
            metadata
        """

        # Generate VBE metadata
        vbe_metadata = self._generate_vbe_metadata(
            offsets, batch_size_per_feature_per_rank
        )

        vbe = vbe_metadata.B_offsets is not None
        # Note this check has already been done in C++ side
        # TODO:  max_B <= self.info_B_mask in python
        # We cannot use assert as it breaks pt2 compile for dynamic shape
        # and need to use torch._check for dynamic shape and cannot construct fstring, use constant string.
        # torch._check(
        #     max_B <= self.info_B_mask,
        #     "Not enough infos bits to accommodate T and B.",
        # )
        # We cannot use lambda as it fails jit script.
        # torch._check is also not supported in jitscript

        # TODO: remove this and add an assert after updating
        # bounds_check_indices to support different indices type and offset
        # type
        force_cast_input_types = (
            indices.dtype != offsets.dtype or force_cast_input_types
        )

        if force_cast_input_types:
            # NOTE: Force offsets to have the same dtype as indices since the
            # kernels assume same dtype.  We might need to revisit the assumption
            # of same dtypes in the future.
            if self.embedding_table_index_type == torch.int32:
                self.log(
                    "Casting indices to int32 based on embedding_table_index_type input."
                )
                indices = indices.to(torch.int32)
            if self.embedding_table_index_type != self.embedding_table_offset_type:
                self.log(
                    f"Force casting offsets to {self.embedding_table_index_type} so that it is the same as the indices type."
                )
            offsets = offsets.to(dtype=indices.dtype)

            # Force casting per_sample_weights to float
            if per_sample_weights is not None:
                per_sample_weights = per_sample_weights.float()

        if self.bounds_check_mode_int != BoundsCheckMode.NONE.value:
            # Override the bounds check version based on prefetch_pipeline
            use_bounds_check_v2 = self.bounds_check_version == 2 or prefetch_pipeline
            bounds_check_version = (
                2 if use_bounds_check_v2 else self.bounds_check_version
            )

            vbe = vbe_metadata.B_offsets is not None

            # Compute B info and VBE metadata for bounds_check_indices only if
            # VBE and bounds check indices v2 are used
            if vbe and use_bounds_check_v2:
                B_offsets = vbe_metadata.B_offsets
                B_offsets_rank_per_feature = vbe_metadata.B_offsets_rank_per_feature
                output_offsets_feature_rank = vbe_metadata.output_offsets_feature_rank
                assert isinstance(B_offsets, Tensor), "B_offsets must be tensor"
                assert isinstance(
                    B_offsets_rank_per_feature, Tensor
                ), "B_offsets_rank_per_feature must be tensor"
                assert isinstance(
                    output_offsets_feature_rank, Tensor
                ), "output_offsets_feature_rank must be tensor"

                row_output_offsets, b_t_map = torch.ops.fbgemm.generate_vbe_metadata(
                    B_offsets,
                    B_offsets_rank_per_feature,
                    output_offsets_feature_rank,
                    self.D_offsets,
                    self.max_D,
                    self.is_nobag,
                    vbe_metadata.max_B_feature_rank,
                    self.info_B_num_bits,
                    offsets.numel() - 1,  # total_B
                )
            else:
                b_t_map = None

            torch.ops.fbgemm.bounds_check_indices(
                self.rows_per_table,
                indices,
                offsets,
                self.bounds_check_mode_int,
                self.bounds_check_warning,
                per_sample_weights,
                B_offsets=vbe_metadata.B_offsets,
                max_B=vbe_metadata.max_B,
                b_t_map=b_t_map,
                info_B_num_bits=self.info_B_num_bits,
                info_B_mask=self.info_B_mask,
                bounds_check_version=bounds_check_version,
                prefetch_pipeline=prefetch_pipeline,
            )

        return indices, offsets, per_sample_weights, vbe_metadata

    def _debug_print_input_stats_factory(self) -> Callable[..., None]:
        """
        If the environment variable FBGEMM_DEBUG_PRINT_INPUT_STATS=1,
        return a function pointer of a function that prints input
        stats including weighted/unweighted, number of features,
        batch size, average pooling factor, total number of indices,
        number of unique indices, and number of indices that goes
        through the different backward functions. Otherwise, return
        a dummy function pointer.
        """

        @torch.jit.ignore
        def _debug_print_input_stats_factory_impl(
            indices: Tensor,
            offsets: Tensor,
            per_sample_weights: Optional[Tensor] = None,
        ) -> None:
            """
            Print input stats (for debugging purpose only)

            Args:
                indices (Tensor): Input indices
                offsets (Tensor): Input offsets
                per_sample_weights (Optional[Tensor]): Input per
                    sample weights
            """
            # pyre-fixme[29]: `Union[(self: TensorBase, other: Union[bool, complex,
            #  float, int, Tensor]) -> Tensor, Module, Tensor]` is not a function.
            if self.debug_step % 100 == 0:
                # Get number of features (T) and batch size (B)
                T = len(self.feature_table_map)
                B = (offsets.numel() - 1) // T

                # Transfer hash_size_cumsum, indices and offsets to CPU
                hash_size_cumsum_cpu = self.hash_size_cumsum.cpu()
                indices_cpu = indices.cpu()
                offsets_cpu = offsets.cpu()

                # Compute linear indices
                for t in range(T):
                    start = offsets_cpu[B * t].item()
                    end = offsets_cpu[B * (t + 1)].item()
                    indices_cpu[start:end] += hash_size_cumsum_cpu[t]

                # Compute unique indices
                uniq_indices_cpu, counts = indices_cpu.unique(return_counts=True)

                # Compute num unique indices
                num_uniq_indices = uniq_indices_cpu.numel()

                # The warp_per_row kernel handles indices that their
                # segment lengths <= 32
                #
                # The cta_per_row kernel handles indices that their
                # segment lengths > 32. A single thread block is used
                # if segment lengths <= 1024. Otherwise, multiple
                # thread blocks are used.
                #
                # Counts of indices that segment lengths <= 32
                counts_warp_per_row = counts[counts <= 32]
                counts_cta_per_row = counts[counts > 32]
                # Counts of indices that segment lengths > 32 and <= 1024
                counts_cta_per_row_sth = counts_cta_per_row[counts_cta_per_row <= 1024]
                # Counts of indices that segment lengths > 1024
                counts_cta_per_row_mth = counts_cta_per_row[counts_cta_per_row > 1024]

                def compute_numel_and_avg(counts: Tensor) -> Tuple[int, float]:
                    numel = counts.numel()
                    avg = (counts.sum().item() / numel) if numel != 0 else -1.0
                    return numel, avg

                # warp_per_row stats
                num_warp_per_row, avg_seglen_warp_per_row = compute_numel_and_avg(
                    counts_warp_per_row
                )
                # cta_per_row using a single thread block stats
                num_cta_per_row_sth, avg_seglen_cta_per_row_sth = compute_numel_and_avg(
                    counts_cta_per_row_sth
                )
                # cta_per_row using multiple thread block stats
                num_cta_per_row_mth, avg_seglen_cta_per_row_mth = compute_numel_and_avg(
                    counts_cta_per_row_mth
                )

                assert num_uniq_indices == (
                    num_warp_per_row + num_cta_per_row_sth + num_cta_per_row_mth
                )

                self.log(
                    "TBE_DEBUG: "
                    "weighted {} "
                    "num features {} "
                    "batch size {} "
                    "avg pooling factor {:.2f} "
                    "total num indices {} "
                    "num unique indices {} "
                    "num warp_per_row {} (avg segment length {:.2f}) "
                    "num cta_per_row single thread block (avg segment length) {} ({:.2f}) "
                    "num cta_per_row multiple thread blocks (avg segment length) {} ({:.2f})".format(
                        per_sample_weights is not None,
                        T,
                        B,
                        indices.numel() / (B * T),
                        indices.numel(),
                        num_uniq_indices,
                        num_warp_per_row,
                        avg_seglen_warp_per_row,
                        num_cta_per_row_sth,
                        avg_seglen_cta_per_row_sth,
                        num_cta_per_row_mth,
                        avg_seglen_cta_per_row_mth,
                    )
                )
            # pyre-fixme[16]: `SplitTableBatchedEmbeddingBagsCodegen` has no
            #  attribute `debug_step`.
            # pyre-fixme[29]: `Union[(self: TensorBase, other: Union[bool, complex,
            #  float, int, Tensor]) -> Tensor, Module, Tensor]` is not a function.
            self.debug_step += 1

        @torch.jit.ignore
        def _debug_print_input_stats_factory_null(
            indices: Tensor,
            offsets: Tensor,
            per_sample_weights: Optional[Tensor] = None,
        ) -> None:
            pass

        if int(os.environ.get("FBGEMM_DEBUG_PRINT_INPUT_STATS", "0")) == 1:
            # pyre-fixme[16]: `SplitTableBatchedEmbeddingBagsCodegen` has no
            #  attribute `debug_step`.
            self.debug_step = 0
            return _debug_print_input_stats_factory_impl
        return _debug_print_input_stats_factory_null


class DenseTableBatchedEmbeddingBagsCodegen(nn.Module):
    """
    Table-batched version of nn.EmbeddingBag(sparse=False)
    """

    weights: Tensor
    weights_offsets: Tensor
    D_offsets: Tensor
    total_D: int
    max_D: int
    hash_size_cumsum: Tensor
    total_hash_size_bits: int
    embedding_specs: List[Tuple[int, int]]

    def __init__(
        self,
        embedding_specs: List[Tuple[int, int]],  # tuple of (rows, dims)
        feature_table_map: Optional[List[int]] = None,  # [T]
        weights_precision: SparseType = SparseType.FP32,
        pooling_mode: PoolingMode = PoolingMode.SUM,
        use_cpu: bool = False,
        output_dtype: SparseType = SparseType.FP32,
        use_mtia: bool = False,
    ) -> None:  # noqa C901  # tuple of (rows, dims,)
        super(DenseTableBatchedEmbeddingBagsCodegen, self).__init__()
        self.uuid = str(uuid.uuid4())

        self.log(
            f"Feature Gates: {[(feature.name, feature.is_enabled()) for feature in FeatureGateName]}"
        )

        self.pooling_mode = pooling_mode
        self.weights_precision = weights_precision
        self.output_dtype: int = output_dtype.as_int()
        table_embedding_dtype = weights_precision.as_dtype()

        self.use_cpu: bool = use_cpu
        self.use_mtia: bool = use_mtia

        assert not (use_cpu and use_mtia), "Cannot use CPU and MTIA at the same time"

        if self.use_cpu or self.pooling_mode == PoolingMode.NONE:
            assert output_dtype in [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.BF16,
            ], "Fused pooled embedding quantization only supported for cuda."

        # pyre-fixme[8]: Attribute has type `device`; used as `Union[int, device]`.
        self.current_device: torch.device = (
            torch.device("cpu")
            if self.use_cpu
            else (
                torch.device(f"mtia:{torch.mtia.current_device()}")
                if self.use_mtia
                else torch.cuda.current_device()
            )
        )

        self.embedding_specs = embedding_specs
        (rows, dims) = zip(*embedding_specs)
        T_ = len(self.embedding_specs)
        assert T_ > 0

        feature_table_map = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        T = len(feature_table_map)
        assert T_ <= T

        feature_dims = [dims[t] for t in feature_table_map]
        D_offsets = [0] + list(accumulate(feature_dims))
        self.total_D = D_offsets[-1]
        self.max_D = max(dims)
        self.register_buffer(
            "D_offsets",
            torch.tensor(D_offsets, device=self.current_device, dtype=torch.int32),
        )
        assert self.D_offsets.numel() == T + 1

        # Required for VBE
        self.register_buffer(
            "feature_dims",
            torch.tensor(feature_dims, device="cpu", dtype=torch.int64),
        )

        hash_size_cumsum = [0] + list(accumulate(rows))
        if hash_size_cumsum[-1] == 0:
            self.total_hash_size_bits: int = 0
        else:
            self.total_hash_size_bits: int = int(log2(float(hash_size_cumsum[-1])) + 1)
        # The last element is to easily access # of rows of each table by
        # hash_size_cumsum[t + 1] - hash_size_cumsum[t]
        hash_size_cumsum = [hash_size_cumsum[t] for t in feature_table_map] + [
            hash_size_cumsum[-1]
        ]
        self.register_buffer(
            "hash_size_cumsum",
            torch.tensor(
                hash_size_cumsum, device=self.current_device, dtype=torch.int64
            ),
        )
        weights_offsets = [0] + list(
            accumulate([row * dim for (row, dim) in embedding_specs])
        )
        self.weights = nn.Parameter(
            torch.randn(
                weights_offsets[-1],
                device=self.current_device,
                dtype=table_embedding_dtype,
            )
        )
        for feature in range(T):
            t = feature_table_map[feature]
            row, dim = embedding_specs[t]
            if (
                self.weights[weights_offsets[t] : weights_offsets[t + 1]].numel()
                != row * dim
            ):
                self.log(
                    f"row {row} dim {dim} feature {feature} t {t} {self.weights[weights_offsets[t] : weights_offsets[t + 1]].numel()}"
                )
            assert (
                self.weights[weights_offsets[t] : weights_offsets[t + 1]].numel()
                == row * dim
            )
            assert self.hash_size_cumsum[feature] == sum(
                row for (row, _) in embedding_specs[:t]
            )

        self.weights_physical_offsets: List[int] = weights_offsets
        weights_offsets = [weights_offsets[t] for t in feature_table_map]
        self.register_buffer(
            "weights_offsets",
            torch.tensor(
                weights_offsets, device=self.current_device, dtype=torch.int64
            ),
        )

    @torch.jit.ignore
    def log(self, msg: str) -> None:
        """
        Log with TBE id prefix to distinguish between multiple TBE instances
        per process

        Args:
            msg (str): The message to print

        Returns:
            None
        """
        logging.info(f"[TBE={self.uuid}] {msg}")

    @torch.jit.ignore
    def _generate_vbe_metadata(
        self,
        offsets: Tensor,
        batch_size_per_feature_per_rank: Optional[List[List[int]]],
    ) -> invokers.lookup_args.VBEMetadata:
        # Blocking D2H copy, but only runs at first call
        self.feature_dims = self.feature_dims.cpu()
        return generate_vbe_metadata(
            offsets,
            batch_size_per_feature_per_rank,
            self.pooling_mode,
            self.feature_dims,
            self.current_device,
        )

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
    ) -> Tensor:
        # Generate VBE metadata
        vbe_metadata = self._generate_vbe_metadata(
            offsets, batch_size_per_feature_per_rank
        )

        # NOTE: Force offsets to have the same dtype as indices since the
        # kernels assume same dtype.  We might need to revisit the assumption
        # of same dtypes in the future.
        offsets = offsets.to(dtype=indices.dtype)

        # Force casting per_sample_weights to float
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.float()

        return torch.ops.fbgemm.dense_embedding_codegen_lookup_function(
            dev_weights=self.weights,
            weights_offsets=self.weights_offsets,
            D_offsets=self.D_offsets,
            total_D=self.total_D,
            max_D=self.max_D,
            hash_size_cumsum=self.hash_size_cumsum,
            total_hash_size_bits=self.total_hash_size_bits,
            indices=indices,
            offsets=offsets,
            pooling_mode=self.pooling_mode,
            indice_weights=per_sample_weights,
            feature_requires_grad=feature_requires_grad,
            output_dtype=self.output_dtype,
            B_offsets=vbe_metadata.B_offsets,
            vbe_output_offsets_feature_rank=vbe_metadata.output_offsets_feature_rank,
            vbe_B_offsets_rank_per_feature=vbe_metadata.B_offsets_rank_per_feature,
            max_B=vbe_metadata.max_B,
            max_B_feature_rank=vbe_metadata.max_B_feature_rank,
            vbe_output_size=vbe_metadata.output_size,
        )

    @torch.jit.export
    def split_embedding_weights(self) -> List[Tensor]:
        """
        Returns a list of weights, split by table
        """
        splits = []
        for t, (rows, dim) in enumerate(self.embedding_specs):
            offset = self.weights_physical_offsets[t]
            splits.append(
                self.weights.detach()[offset : offset + rows * dim].view(rows, dim)
            )
        return splits

    def init_embedding_weights_uniform(self, min_val: float, max_val: float) -> None:
        splits = self.split_embedding_weights()
        for param in splits:
            param.uniform_(min_val, max_val)
