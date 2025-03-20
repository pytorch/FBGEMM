#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[13,56]

import contextlib
import functools
import itertools
import logging
import os
import tempfile
import threading
import time
from math import floor, log2
from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch  # usort:skip

# @manual=//deeplearning/fbgemm/fbgemm_gpu/codegen:split_embedding_codegen_lookup_invokers
import fbgemm_gpu.split_embedding_codegen_lookup_invokers as invokers
from fbgemm_gpu.runtime_monitor import (
    AsyncSeriesTimer,
    TBEStatsReporter,
    TBEStatsReporterConfig,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
    SplitState,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    apply_split_helper,
    CounterBasedRegularizationDefinition,
    CowClipDefinition,
    UVMCacheStatsIndex,
    WeightDecayMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import (
    generate_vbe_metadata,
)

from torch import distributed as dist, nn, Tensor  # usort:skip
from dataclasses import dataclass

from torch.autograd.profiler import record_function

from ..cache import get_unique_indices_v2

from .common import ASSOC
from .utils.partially_materialized_tensor import PartiallyMaterializedTensor


@dataclass
class IterData:
    indices: Tensor
    offsets: Tensor
    lxu_cache_locations: Tensor
    lxu_cache_ptrs: Tensor
    actions_count_gpu: Tensor
    cache_set_inverse_indices: Tensor
    B_offsets: Optional[Tensor] = None
    max_B: Optional[int] = -1


class SSDTableBatchedEmbeddingBags(nn.Module):
    D_offsets: Tensor
    lxu_cache_weights: Tensor
    lru_state: Tensor
    lxu_cache_weights: Tensor
    lxu_cache_state: Tensor
    momentum1_dev: Tensor
    momentum1_uvm: Tensor
    momentum1_host: Tensor
    momentum1_placements: Tensor
    momentum1_offsets: Tensor
    weights_dev: Tensor
    weights_uvm: Tensor
    weights_host: Tensor
    weights_placements: Tensor
    weights_offsets: Tensor
    _local_instance_index: int = -1

    def __init__(
        self,
        embedding_specs: List[Tuple[int, int]],  # tuple of (rows, dims)
        feature_table_map: Optional[List[int]],  # [T]
        cache_sets: int,
        ssd_storage_directory: str,
        ssd_rocksdb_shards: int = 1,
        ssd_memtable_flush_period: int = -1,
        ssd_memtable_flush_offset: int = -1,
        ssd_l0_files_per_compact: int = 4,
        ssd_rate_limit_mbps: int = 0,
        ssd_size_ratio: int = 10,
        ssd_compaction_trigger: int = 8,
        ssd_rocksdb_write_buffer_size: int = 2 * 1024 * 1024 * 1024,
        ssd_max_write_buffer_num: int = 4,
        ssd_cache_location: EmbeddingLocation = EmbeddingLocation.MANAGED,
        ssd_uniform_init_lower: float = -0.01,
        ssd_uniform_init_upper: float = 0.01,
        ssd_block_cache_size_per_tbe: int = 0,
        weights_precision: SparseType = SparseType.FP32,
        output_dtype: SparseType = SparseType.FP32,
        optimizer: OptimType = OptimType.EXACT_ROWWISE_ADAGRAD,
        # General Optimizer args
        stochastic_rounding: bool = True,
        gradient_clipping: bool = False,
        max_gradient: float = 1.0,
        max_norm: float = 0.0,
        learning_rate: float = 0.01,
        eps: float = 1.0e-8,  # used by Adagrad, LAMB, and Adam
        momentum: float = 0.9,  # used by LARS-SGD
        weight_decay: float = 0.0,  # used by LARS-SGD, LAMB, ADAM, and Rowwise Adagrad
        weight_decay_mode: WeightDecayMode = WeightDecayMode.NONE,  # used by Rowwise Adagrad
        eta: float = 0.001,  # used by LARS-SGD,
        beta1: float = 0.9,  # used by LAMB and ADAM
        beta2: float = 0.999,  # used by LAMB and ADAM
        counter_based_regularization: Optional[
            CounterBasedRegularizationDefinition
        ] = None,  # used by Rowwise Adagrad
        cowclip_regularization: Optional[
            CowClipDefinition
        ] = None,  # used by Rowwise Adagrad
        pooling_mode: PoolingMode = PoolingMode.SUM,
        bounds_check_mode: BoundsCheckMode = BoundsCheckMode.WARNING,
        # Parameter Server Configs
        ps_hosts: Optional[Tuple[Tuple[str, int]]] = None,
        ps_max_key_per_request: Optional[int] = None,
        ps_client_thread_num: Optional[int] = None,
        ps_max_local_index_length: Optional[int] = None,
        tbe_unique_id: int = -1,
        # in local test we need to use the pass in path for rocksdb creation
        # in production we need to do it inside SSD mount path which will ignores the passed in path
        use_passed_in_path: int = True,
        gather_ssd_cache_stats: Optional[bool] = False,
        stats_reporter_config: Optional[TBEStatsReporterConfig] = None,
        l2_cache_size: int = 0,
        # Set to True to enable pipeline prefetching
        prefetch_pipeline: bool = False,
        # Set to True to alloc a UVM tensor using malloc+cudaHostRegister.
        # Set to False to use cudaMallocManaged
        uvm_host_mapped: bool = False,
        enable_async_update: bool = True,  # whether enable L2/rocksdb write to async background thread
        # if > 0, insert all kv pairs to rocksdb at init time, in chunks of *bulk_init_chunk_size* bytes
        # number of rows will be decided by bulk_init_chunk_size / size_of_each_row
        bulk_init_chunk_size: int = 0,
        lazy_bulk_init_enabled: bool = False,
    ) -> None:
        super(SSDTableBatchedEmbeddingBags, self).__init__()

        self.pooling_mode = pooling_mode
        self.bounds_check_mode_int: int = bounds_check_mode.value
        self.embedding_specs = embedding_specs
        (rows, dims) = zip(*embedding_specs)
        T_ = len(self.embedding_specs)
        assert T_ > 0
        # pyre-fixme[8]: Attribute has type `device`; used as `int`.
        self.current_device: torch.device = torch.cuda.current_device()

        self.feature_table_map: List[int] = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        T = len(self.feature_table_map)
        assert T_ <= T
        table_has_feature = [False] * T_
        for t in self.feature_table_map:
            table_has_feature[t] = True
        assert all(table_has_feature), "Each table must have at least one feature!"

        feature_dims = [dims[t] for t in self.feature_table_map]
        D_offsets = [dims[t] for t in self.feature_table_map]
        D_offsets = [0] + list(itertools.accumulate(D_offsets))
        self.total_D: int = D_offsets[-1]
        self.max_D: int = max(dims)
        self.register_buffer(
            "D_offsets",
            torch.tensor(D_offsets, device=self.current_device, dtype=torch.int32),
        )
        assert self.D_offsets.numel() == T + 1
        hash_size_cumsum = [0] + list(itertools.accumulate(rows))
        if hash_size_cumsum[-1] == 0:
            self.total_hash_size_bits: int = 0
        else:
            self.total_hash_size_bits: int = int(log2(float(hash_size_cumsum[-1])) + 1)
        # The last element is to easily access # of rows of each table by
        self.total_hash_size_bits = int(log2(float(hash_size_cumsum[-1])) + 1)
        self.total_hash_size: int = hash_size_cumsum[-1]
        # The last element is to easily access # of rows of each table by
        # hash_size_cumsum[t + 1] - hash_size_cumsum[t]
        hash_size_cumsum = [hash_size_cumsum[t] for t in self.feature_table_map] + [
            hash_size_cumsum[-1]
        ]
        self.register_buffer(
            "hash_size_cumsum",
            torch.tensor(
                hash_size_cumsum, device=self.current_device, dtype=torch.int64
            ),
        )

        self.uvm_host_mapped = uvm_host_mapped
        logging.info(
            f"TBE will allocate a UVM buffer with is_host_mapped={uvm_host_mapped}"
        )
        self.bulk_init_chunk_size = bulk_init_chunk_size
        self.lazy_init_thread: threading.Thread | None = None

        # Buffers for bounds check
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

        assert cache_sets > 0
        element_size = weights_precision.bit_rate() // 8
        assert (
            element_size == 4 or element_size == 2
        ), f"Invalid element size {element_size}"
        cache_size = cache_sets * ASSOC * element_size * self.max_D
        logging.info(
            f"Using cache for SSD with admission algorithm "
            f"{CacheAlgorithm.LRU}, {cache_sets} sets, stored on {'DEVICE' if ssd_cache_location is EmbeddingLocation.DEVICE else 'MANAGED'} with {ssd_rocksdb_shards} shards, "
            f"SSD storage directory: {ssd_storage_directory}, "
            f"Memtable Flush Period: {ssd_memtable_flush_period}, "
            f"Memtable Flush Offset: {ssd_memtable_flush_offset}, "
            f"Desired L0 files per compaction: {ssd_l0_files_per_compact}, "
            f"{cache_size / 1024.0 / 1024.0 / 1024.0 : .2f}GB, "
            f"weights precision: {weights_precision}, "
            f"output dtype: {output_dtype}, "
            f"chunk size in bulk init: {bulk_init_chunk_size} bytes"
        )
        self.register_buffer(
            "lxu_cache_state",
            torch.zeros(
                cache_sets, ASSOC, device=self.current_device, dtype=torch.int64
            ).fill_(-1),
        )
        self.register_buffer(
            "lru_state",
            torch.zeros(
                cache_sets, ASSOC, device=self.current_device, dtype=torch.int64
            ),
        )

        self.step = 0

        # Set prefetch pipeline
        self.prefetch_pipeline: bool = prefetch_pipeline
        self.prefetch_stream: Optional[torch.cuda.Stream] = None

        # Cache locking counter for pipeline prefetching
        if self.prefetch_pipeline:
            self.register_buffer(
                "lxu_cache_locking_counter",
                torch.zeros(
                    cache_sets,
                    ASSOC,
                    device=self.current_device,
                    dtype=torch.int32,
                ),
                persistent=True,
            )
        else:
            self.register_buffer(
                "lxu_cache_locking_counter",
                torch.zeros([0, 0], dtype=torch.int32, device=self.current_device),
                persistent=False,
            )

        assert ssd_cache_location in (
            EmbeddingLocation.MANAGED,
            EmbeddingLocation.DEVICE,
        )

        assert weights_precision in (SparseType.FP32, SparseType.FP16)
        self.weights_precision = weights_precision
        self.output_dtype: int = output_dtype.as_int()

        cache_dtype = weights_precision.as_dtype()
        if ssd_cache_location == EmbeddingLocation.MANAGED:
            self.register_buffer(
                "lxu_cache_weights",
                torch.ops.fbgemm.new_unified_tensor(
                    torch.zeros(
                        1,
                        device=self.current_device,
                        dtype=cache_dtype,
                    ),
                    [cache_sets * ASSOC, self.max_D],
                    is_host_mapped=self.uvm_host_mapped,
                ),
            )
        else:
            self.register_buffer(
                "lxu_cache_weights",
                torch.zeros(
                    cache_sets * ASSOC,
                    self.max_D,
                    device=self.current_device,
                    dtype=cache_dtype,
                ),
            )
            assert (
                cache_size
                == self.lxu_cache_weights.numel()
                * self.lxu_cache_weights.element_size()
            ), "The precomputed cache_size does not match the actual cache size"

        # Buffers for cache eviction
        # For storing weights to evict
        # The max number of rows to be evicted is limited by the number of
        # slots in the cache. Thus, we allocate `lxu_cache_evicted_weights` to
        # be the same shape as the L1 cache (lxu_cache_weights)
        self.register_buffer(
            "lxu_cache_evicted_weights",
            torch.ops.fbgemm.new_unified_tensor(
                torch.zeros(
                    1,
                    device=self.current_device,
                    dtype=cache_dtype,
                ),
                self.lxu_cache_weights.shape,
                is_host_mapped=self.uvm_host_mapped,
            ),
        )

        # For storing embedding indices to evict to
        self.register_buffer(
            "lxu_cache_evicted_indices",
            torch.ops.fbgemm.new_unified_tensor(
                torch.zeros(
                    1,
                    device=self.current_device,
                    dtype=torch.long,
                ),
                (self.lxu_cache_weights.shape[0],),
                is_host_mapped=self.uvm_host_mapped,
            ),
        )

        # For storing cache slots to evict
        self.register_buffer(
            "lxu_cache_evicted_slots",
            torch.ops.fbgemm.new_unified_tensor(
                torch.zeros(
                    1,
                    device=self.current_device,
                    dtype=torch.int,
                ),
                (self.lxu_cache_weights.shape[0],),
                is_host_mapped=self.uvm_host_mapped,
            ),
        )

        # For storing the number of evicted rows
        self.register_buffer(
            "lxu_cache_evicted_count",
            torch.ops.fbgemm.new_unified_tensor(
                torch.zeros(
                    1,
                    device=self.current_device,
                    dtype=torch.int,
                ),
                (1,),
                is_host_mapped=self.uvm_host_mapped,
            ),
        )

        self.timestep = 0

        # Dummy profile configuration for measuring the SSD get/set time
        # get and set are executed by another thread which (for some reason) is
        # not traceable by PyTorch's Kineto. We workaround this problem by
        # injecting a dummy kernel into the GPU stream to make it traceable
        #
        # This function can be enabled by setting an environment variable
        # FBGEMM_SSD_TBE_USE_DUMMY_PROFILE=1
        self.dummy_profile_tensor: Tensor = torch.as_tensor(
            [0], device=self.current_device, dtype=torch.int
        )
        set_dummy_profile = os.environ.get("FBGEMM_SSD_TBE_USE_DUMMY_PROFILE")
        use_dummy_profile = False
        if set_dummy_profile is not None:
            use_dummy_profile = int(set_dummy_profile) == 1
            logging.info(
                f"FBGEMM_SSD_TBE_USE_DUMMY_PROFILE is set to {set_dummy_profile}; "
                f"Use dummy profile: {use_dummy_profile}"
            )
        # pyre-ignore[4]
        self.record_function_via_dummy_profile: Callable[..., Any] = (
            self.record_function_via_dummy_profile_factory(use_dummy_profile)
        )

        os.makedirs(ssd_storage_directory, exist_ok=True)

        ssd_directory = tempfile.mkdtemp(
            prefix="ssd_table_batched_embeddings", dir=ssd_storage_directory
        )
        # logging.info("DEBUG: weights_precision {}".format(weights_precision))

        # create tbe unique id using rank index | local tbe idx
        if tbe_unique_id == -1:
            SSDTableBatchedEmbeddingBags._local_instance_index += 1
            if dist.is_initialized():
                assert (
                    SSDTableBatchedEmbeddingBags._local_instance_index < 1024
                ), f"{SSDTableBatchedEmbeddingBags._local_instance_index}, more than 1024 TBE instance is created in one rank, the tbe unique id won't be unique in this case."
                tbe_unique_id = (
                    dist.get_rank() << 10
                    | SSDTableBatchedEmbeddingBags._local_instance_index
                )
            else:
                logging.warning("dist is not initialized, treating as single gpu cases")
                tbe_unique_id = SSDTableBatchedEmbeddingBags._local_instance_index
        self.tbe_unique_id = tbe_unique_id
        logging.info(f"tbe_unique_id: {tbe_unique_id}")
        if not ps_hosts:
            logging.info(
                f"Logging SSD offloading setup, tbe_unique_id:{tbe_unique_id}, l2_cache_size:{l2_cache_size}GB, enable_async_update:{enable_async_update}"
                f"passed_in_path={ssd_directory}, num_shards={ssd_rocksdb_shards},num_threads={ssd_rocksdb_shards},"
                f"memtable_flush_period={ssd_memtable_flush_period},memtable_flush_offset={ssd_memtable_flush_offset},"
                f"l0_files_per_compact={ssd_l0_files_per_compact},max_D={self.max_D},rate_limit_mbps={ssd_rate_limit_mbps},"
                f"size_ratio={ssd_size_ratio},compaction_trigger={ssd_compaction_trigger}, lazy_bulk_init_enabled={lazy_bulk_init_enabled},"
                f"write_buffer_size_per_tbe={ssd_rocksdb_write_buffer_size},max_write_buffer_num_per_db_shard={ssd_max_write_buffer_num},"
                f"uniform_init_lower={ssd_uniform_init_lower},uniform_init_upper={ssd_uniform_init_upper},"
                f"row_storage_bitwidth={weights_precision.bit_rate()},block_cache_size_per_tbe={ssd_block_cache_size_per_tbe},"
                f"use_passed_in_path:{use_passed_in_path}, real_path will be printed in EmbeddingRocksDB"
            )
            # pyre-fixme[4]: Attribute must be annotated.
            # pyre-ignore[16]
            self._ssd_db = torch.classes.fbgemm.EmbeddingRocksDBWrapper(
                ssd_directory,
                ssd_rocksdb_shards,
                ssd_rocksdb_shards,
                ssd_memtable_flush_period,
                ssd_memtable_flush_offset,
                ssd_l0_files_per_compact,
                self.max_D,
                ssd_rate_limit_mbps,
                ssd_size_ratio,
                ssd_compaction_trigger,
                ssd_rocksdb_write_buffer_size,
                ssd_max_write_buffer_num,
                ssd_uniform_init_lower,
                ssd_uniform_init_upper,
                weights_precision.bit_rate(),  # row_storage_bitwidth
                ssd_block_cache_size_per_tbe,
                use_passed_in_path,
                tbe_unique_id,
                l2_cache_size,
                enable_async_update,
            )
            if self.bulk_init_chunk_size > 0:
                self.ssd_uniform_init_lower: float = ssd_uniform_init_lower
                self.ssd_uniform_init_upper: float = ssd_uniform_init_upper
                if lazy_bulk_init_enabled:
                    self._lazy_initialize_ssd_tbe()
                else:
                    self._insert_all_kv()
        else:
            # pyre-fixme[4]: Attribute must be annotated.
            # pyre-ignore[16]
            self._ssd_db = torch.classes.fbgemm.EmbeddingParameterServerWrapper(
                [host[0] for host in ps_hosts],
                [host[1] for host in ps_hosts],
                tbe_unique_id,
                (
                    ps_max_local_index_length
                    if ps_max_local_index_length is not None
                    else 54
                ),
                ps_client_thread_num if ps_client_thread_num is not None else 32,
                ps_max_key_per_request if ps_max_key_per_request is not None else 500,
                l2_cache_size,
                self.max_D,
            )
        # pyre-fixme[20]: Argument `self` expected.
        (low_priority, high_priority) = torch.cuda.Stream.priority_range()
        # GPU stream for SSD cache eviction
        self.ssd_eviction_stream = torch.cuda.Stream(priority=low_priority)
        # GPU stream for SSD memory copy
        self.ssd_memcpy_stream = torch.cuda.Stream(priority=low_priority)

        # SSD get completion event
        self.ssd_event_get = torch.cuda.Event()
        # SSD scratch pad eviction completion event
        self.ssd_event_sp_evict = torch.cuda.Event()
        # SSD cache eviction completion event
        self.ssd_event_cache_evict = torch.cuda.Event()
        # SSD backward completion event
        self.ssd_event_backward = torch.cuda.Event()
        # SSD get's input copy completion event
        self.ssd_event_get_inputs_cpy = torch.cuda.Event()
        if self.prefetch_pipeline:
            # SSD scratch pad index queue insert completion event
            self.ssd_event_sp_idxq_insert: torch.cuda.streams.Event = torch.cuda.Event()
            # SSD scratch pad index queue lookup completion event
            self.ssd_event_sp_idxq_lookup: torch.cuda.streams.Event = torch.cuda.Event()

        self.timesteps_prefetched: List[int] = []
        # TODO: add type annotation
        # pyre-fixme[4]: Attribute must be annotated.
        self.ssd_prefetch_data = []

        # Scratch pad eviction data queue
        self.ssd_scratch_pad_eviction_data: List[
            Tuple[Tensor, Tensor, Tensor, bool]
        ] = []
        self.ssd_location_update_data: List[Tuple[Tensor, Tensor]] = []

        if self.prefetch_pipeline:
            # Scratch pad value queue
            self.ssd_scratch_pads: List[Tuple[Tensor, Tensor, Tensor]] = []

            # pyre-ignore[4]
            # Scratch pad index queue
            self.scratch_pad_idx_queue = torch.classes.fbgemm.SSDScratchPadIndicesQueue(
                -1
            )

        if weight_decay_mode == WeightDecayMode.COUNTER or counter_based_regularization:
            raise AssertionError(
                "weight_decay_mode = WeightDecayMode.COUNTER is not supported for SSD TBE."
            )
        counter_based_regularization = CounterBasedRegularizationDefinition()

        if weight_decay_mode == WeightDecayMode.COWCLIP or cowclip_regularization:
            raise AssertionError(
                "weight_decay_mode = WeightDecayMode.COWCLIP is not supported for SSD TBE."
            )
        cowclip_regularization = CowClipDefinition()

        self.optimizer_args = invokers.lookup_args_ssd.OptimizerArgs(
            stochastic_rounding=stochastic_rounding,
            gradient_clipping=gradient_clipping,
            max_gradient=max_gradient,
            max_norm=max_norm,
            learning_rate=learning_rate,
            eps=eps,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            weight_decay_mode=weight_decay_mode.value,
            eta=eta,
            momentum=momentum,
            counter_halflife=counter_based_regularization.counter_halflife,
            adjustment_iter=counter_based_regularization.adjustment_iter,
            adjustment_ub=counter_based_regularization.adjustment_ub,
            learning_rate_mode=counter_based_regularization.learning_rate_mode.value,
            grad_sum_decay=counter_based_regularization.grad_sum_decay.value,
            tail_id_threshold=counter_based_regularization.tail_id_threshold.val,
            is_tail_id_thresh_ratio=int(
                counter_based_regularization.tail_id_threshold.is_ratio
            ),
            total_hash_size=-1,  # Unused
            weight_norm_coefficient=cowclip_regularization.weight_norm_coefficient,
            lower_bound=cowclip_regularization.lower_bound,
            regularization_mode=weight_decay_mode.value,
            use_rowwise_bias_correction=False,  # Unused, this is used in TBE's Adam
        )

        table_embedding_dtype = weights_precision.as_dtype()

        self._apply_split(
            SplitState(
                dev_size=0,
                host_size=0,
                uvm_size=0,
                placements=[EmbeddingLocation.MANAGED_CACHING for _ in range(T_)],
                offsets=[0] * (len(rows)),
            ),
            "weights",
            # pyre-fixme[6]: For 3rd argument expected `Type[dtype]` but got `dtype`.
            dtype=table_embedding_dtype,
        )

        momentum1_offsets = [0] + list(itertools.accumulate(rows))
        self._apply_split(
            SplitState(
                dev_size=self.total_hash_size,
                host_size=0,
                uvm_size=0,
                placements=[EmbeddingLocation.DEVICE for _ in range(T_)],
                offsets=momentum1_offsets[:-1],
            ),
            "momentum1",
            # pyre-fixme[6]: For 3rd argument expected `Type[dtype]` but got `dtype`.
            dtype=torch.float32,
        )

        # For storing current iteration data
        self.current_iter_data: Optional[IterData] = None

        # add placeholder require_grad param to enable autograd without nn.parameter
        # this is needed to enable int8 embedding weights for SplitTableBatchedEmbedding
        self.placeholder_autograd_tensor = nn.Parameter(
            torch.zeros(0, device=self.current_device, dtype=torch.float)
        )

        # Register backward hook for evicting rows from a scratch pad to SSD
        # post backward
        self.placeholder_autograd_tensor.register_hook(self._evict_from_scratch_pad)

        if self.prefetch_pipeline:
            self.register_full_backward_pre_hook(
                self._update_cache_counter_and_pointers
            )

        assert optimizer in (
            OptimType.EXACT_ROWWISE_ADAGRAD,
        ), f"Optimizer {optimizer} is not supported by SSDTableBatchedEmbeddingBags"
        self.optimizer = optimizer

        # stats reporter
        self.gather_ssd_cache_stats = gather_ssd_cache_stats
        self.stats_reporter: Optional[TBEStatsReporter] = (
            stats_reporter_config.create_reporter() if stats_reporter_config else None
        )
        self.ssd_cache_stats_size = 6
        # 0: N_calls, 1: N_requested_indices, 2: N_unique_indices, 3: N_unique_misses,
        # 4: N_conflict_unique_misses, 5: N_conflict_misses
        self.last_reported_ssd_stats: List[float] = []
        self.last_reported_step = 0

        self.register_buffer(
            "ssd_cache_stats",
            torch.zeros(
                size=(self.ssd_cache_stats_size,),
                device=self.current_device,
                dtype=torch.int64,
            ),
        )

        self.register_buffer(
            "local_ssd_cache_stats",
            torch.zeros(
                self.ssd_cache_stats_size,
                device=self.current_device,
                dtype=torch.int32,
            ),
        )
        logging.info(
            f"logging stats reporter setup, {self.gather_ssd_cache_stats=}, "
            f"stats_reporter:{self.stats_reporter if self.stats_reporter else 'none'}"
        )

        # prefetch launch a series of kernels, we use AsyncSeriesTimer to track the kernel time
        self.ssd_prefetch_read_timer: Optional[AsyncSeriesTimer] = None
        self.ssd_prefetch_evict_timer: Optional[AsyncSeriesTimer] = None
        self.prefetch_parallel_stream_cnt: int = 2
        # tuple of iteration, prefetch parallel stream cnt, reported duration
        # since there are 2 stream in parallel in prefetch, we want to count the longest one
        self.prefetch_duration_us: Tuple[int, int, float] = (
            -1,
            self.prefetch_parallel_stream_cnt,
            0,
        )
        self.l2_num_cache_misses_stats_name: str = (
            f"l2_cache.perf.get.tbe_id{tbe_unique_id}.num_cache_misses"
        )
        self.l2_num_cache_lookups_stats_name: str = (
            f"l2_cache.perf.get.tbe_id{tbe_unique_id}.num_lookups"
        )
        self.l2_num_cache_evictions_stats_name: str = (
            f"l2_cache.perf.tbe_id{tbe_unique_id}.num_l2_cache_evictions"
        )
        self.l2_cache_free_mem_stats_name: str = (
            f"l2_cache.mem.tbe_id{tbe_unique_id}.free_mem_bytes"
        )
        self.l2_cache_capacity_stats_name: str = (
            f"l2_cache.mem.tbe_id{tbe_unique_id}.capacity_bytes"
        )
        if self.stats_reporter:
            self.ssd_prefetch_read_timer = AsyncSeriesTimer(
                functools.partial(
                    SSDTableBatchedEmbeddingBags._report_duration,
                    self,
                    event_name="tbe.prefetch_duration_us",
                    time_unit="us",
                )
            )
            self.ssd_prefetch_evict_timer = AsyncSeriesTimer(
                functools.partial(
                    SSDTableBatchedEmbeddingBags._report_duration,
                    self,
                    event_name="tbe.prefetch_duration_us",
                    time_unit="us",
                )
            )
            # pyre-ignore
            self.stats_reporter.register_stats(self.l2_num_cache_misses_stats_name)
            # pyre-ignore
            self.stats_reporter.register_stats(self.l2_num_cache_lookups_stats_name)
            self.stats_reporter.register_stats(self.l2_num_cache_evictions_stats_name)
            self.stats_reporter.register_stats(self.l2_cache_free_mem_stats_name)
            self.stats_reporter.register_stats(self.l2_cache_capacity_stats_name)

    @property
    # pyre-ignore
    def ssd_db(self):
        """Intercept the ssd_db property to make sure it is fully initialized before use.
        This is needed because random weights are initialized in a separate thread"""
        if self.lazy_init_thread is not None:
            self.lazy_init_thread.join()
            self.lazy_init_thread = None
            logging.info("lazy ssd tbe initialization completed, weights are ready")

        return self._ssd_db

    @ssd_db.setter
    # pyre-ignore
    def ssd_db(self, value):
        """Setter for ssd_db property."""
        if self.lazy_init_thread is not None:
            # This is essentially a copy assignment operation, since the thread is
            # already existing, and we are assigning a new ssd_db to it. Complete
            # the initialization first, then assign the new value to it.
            self.lazy_init_thread.join()
            self.lazy_init_thread = None
            logging.info(
                "lazy ssd tbe initialization completed, ssd_db will now get overridden"
            )

        self._ssd_db = value

    def _lazy_initialize_ssd_tbe(self) -> None:
        """
        Initialize the SSD TBE with random weights. This function should only be
        called once at initialization time.
        """
        if self.bulk_init_chunk_size > 0:
            self.lazy_init_thread = threading.Thread(target=self._insert_all_kv)
            # pyre-ignore
            self.lazy_init_thread.start()
            logging.info(
                f"lazy ssd tbe initialization started since bulk_init_chunk_size is set to {self.bulk_init_chunk_size}"
            )
        else:
            logging.debug(
                "bulk_init_chunk_size is not set, skipping lazy initialization"
            )

    @torch.jit.ignore
    def _insert_all_kv(self) -> None:
        """
        Populate all rows in the ssd TBE with random weights. Existing keys will
        be effectively overwritten. This function should only be called once at
        initailization time.
        """
        row_offset = 0
        row_count = floor(
            self.bulk_init_chunk_size
            / (self.max_D * self.weights_precision.as_dtype().itemsize)
        )
        total_dim0 = 0
        for dim0, _ in self.embedding_specs:
            total_dim0 += dim0

        start_ts = time.time()
        chunk_tensor = torch.empty(
            row_count,
            self.max_D,
            dtype=self.weights_precision.as_dtype(),
            device="cuda",
        )
        cpu_tensor = torch.empty_like(chunk_tensor, device="cpu")
        for row_offset in range(0, total_dim0, row_count):
            actual_dim0 = min(total_dim0 - row_offset, row_count)
            chunk_tensor.uniform_(
                self.ssd_uniform_init_lower, self.ssd_uniform_init_upper
            )
            cpu_tensor.copy_(chunk_tensor, non_blocking=False)
            rand_val = cpu_tensor[:actual_dim0, :]
            # This code is intentionally not calling through the getter property
            # to avoid the lazy initialization thread from joining with itself.
            self._ssd_db.set_range_to_storage(rand_val, row_offset, actual_dim0)
        self.ssd_db.toggle_compaction(True)
        end_ts = time.time()
        elapsed = int((end_ts - start_ts) * 1e6)
        logging.info(
            f"TBE bulk initialization took {elapsed:_} us, bulk_init_chunk_size={self.bulk_init_chunk_size}, each batch of {row_count} rows, total rows of {total_dim0}"
        )

    @torch.jit.ignore
    def _report_duration(
        self,
        it_step: int,
        dur_ms: float,
        event_name: str,
        time_unit: str,
    ) -> None:
        """
        Callback function passed into AsyncSeriesTimer, which will be
        invoked when the last kernel in AsyncSeriesTimer scope is done.
        Currently this is only used to trace prefetch duration, in which
        there are 2 streams involved, main stream and eviction stream.
        This will report the duration of the longer stream to ODS

        Function is not thread safe

        Args:
            it_step (int): The reporting iteration step
            dur_ms (float): The duration of the all the kernels within the
                            AsyncSeriesTimer scope in milliseconds
            event_name (str): The name of the event
            time_unit (str): The unit of the duration(us or ms)
        """
        recorded_itr, stream_cnt, report_val = self.prefetch_duration_us
        duration = dur_ms
        if time_unit == "us":  # pyre-ignore
            duration = dur_ms * 1000
        if it_step == recorded_itr:
            report_val = max(report_val, duration)
            stream_cnt -= 1
        else:
            # reset
            recorded_itr = it_step
            report_val = duration
            stream_cnt = self.prefetch_parallel_stream_cnt
        self.prefetch_duration_us = (recorded_itr, stream_cnt, report_val)

        if stream_cnt == 1:
            # this is the last stream, handling ods report
            # pyre-ignore
            self.stats_reporter.report_duration(
                it_step, event_name, report_val, time_unit=time_unit
            )

    # pyre-ignore[3]
    def record_function_via_dummy_profile_factory(
        self,
        use_dummy_profile: bool,
    ) -> Callable[..., Any]:
        """
        Generate the record_function_via_dummy_profile based on the
        use_dummy_profile flag.

        If use_dummy_profile is True, inject a dummy kernel before and after
        the function call and record function via `record_function`

        Otherwise, just execute the function

        Args:
            use_dummy_profile (bool): A flag for enabling/disabling
                                      record_function_via_dummy_profile
        """
        if use_dummy_profile:

            def func(
                name: str,
                # pyre-ignore[2]
                fn: Callable[..., Any],
                *args: Any,
                **kwargs: Any,
            ) -> None:
                with record_function(name):
                    self.dummy_profile_tensor.add_(1)
                    fn(*args, **kwargs)
                    self.dummy_profile_tensor.add_(1)

            return func

        def func(
            name: str,
            # pyre-ignore[2]
            fn: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
        ) -> None:
            fn(*args, **kwargs)

        return func

    def _apply_split(
        self,
        split: SplitState,
        prefix: str,
        dtype: Type[torch.dtype],
        enforce_hbm: bool = False,
        make_dev_param: bool = False,
        dev_reshape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        apply_split_helper(
            self.register_buffer,
            functools.partial(setattr, self),
            self.current_device,
            False,  # use_cpu
            self.feature_table_map,
            split,
            prefix,
            dtype,
            enforce_hbm,
            make_dev_param,
            dev_reshape,
        )

    def to_pinned_cpu(self, t: torch.Tensor) -> torch.Tensor:
        t_cpu = torch.empty(t.shape, pin_memory=True, dtype=t.dtype)
        t_cpu.copy_(t, non_blocking=True)
        return t_cpu

    def to_pinned_cpu_on_stream_wait_on_another_stream(
        self,
        tensors: List[Tensor],
        stream: torch.cuda.Stream,
        stream_to_wait_on: torch.cuda.Stream,
        post_event: Optional[torch.cuda.Event] = None,
    ) -> List[Tensor]:
        """
        Transfer input tensors from GPU to CPU using a pinned host
        buffer.  The transfer is carried out on the given stream
        (`stream`) after all the kernels in the other stream
        (`stream_to_wait_on`) are complete.

        Args:
            tensors (List[Tensor]): The list of tensors to be
                                    transferred
            stream (Stream): The stream to run memory copy
            stream_to_wait_on (Stream): The stream to wait on
            post_event (Event): The post completion event

        Returns:
            The list of pinned CPU tensors
        """
        with torch.cuda.stream(stream):
            stream.wait_stream(stream_to_wait_on)
            cpu_tensors = []
            for t in tensors:
                t.record_stream(stream)
                cpu_tensors.append(self.to_pinned_cpu(t))
            if post_event is not None:
                stream.record_event(post_event)
            return cpu_tensors

    def evict(
        self,
        rows: Tensor,
        indices_cpu: Tensor,
        actions_count_cpu: Tensor,
        stream: torch.cuda.Stream,
        pre_event: Optional[torch.cuda.Event],
        post_event: Optional[torch.cuda.Event],
        is_rows_uvm: bool,
        name: Optional[str] = "",
        is_bwd: bool = True,
    ) -> None:
        """
        Evict data from the given input tensors to SSD via RocksDB
        Args:
            rows (Tensor): The 2D tensor that contains rows to evict
            indices_cpu (Tensor): The 1D CPU tensor that contains the row
                                  indices that the rows will be evicted to
            actions_count_cpu (Tensor): A scalar tensor that contains the
                                        number of rows that the evict function
                                        has to process
            stream (Stream): The CUDA stream that cudaStreamAddCallback will
                             synchronize the host function with.  Moreover, the
                             asynchronous D->H memory copies will operate on
                             this stream
            pre_event (Event): The CUDA event that the stream has to wait on
            post_event (Event): The CUDA event that the current will record on
                                when the eviction is done
            is_rows_uvm (bool): A flag to indicate whether `rows` is a UVM
                                tensor (which is accessible on both host and
                                device)
        Returns:
            None
        """
        with record_function(f"## ssd_evict_{name} ##"):
            with torch.cuda.stream(stream):
                if pre_event is not None:
                    stream.wait_event(pre_event)

                rows_cpu = rows if is_rows_uvm else self.to_pinned_cpu(rows)

                rows.record_stream(stream)

                self.record_function_via_dummy_profile(
                    f"## ssd_set_{name} ##",
                    self.ssd_db.set_cuda,
                    indices_cpu,
                    rows_cpu,
                    actions_count_cpu,
                    self.timestep,
                    is_bwd,
                )

                if post_event is not None:
                    stream.record_event(post_event)

    def _evict_from_scratch_pad(self, grad: Tensor) -> None:
        """
        Evict conflict missed rows from a scratch pad
        (`inserted_rows`) on the `ssd_eviction_stream`. This is a hook
        that is invoked right after TBE backward.

        Conflict missed indices are specified in
        `post_bwd_evicted_indices_cpu`. Indices that are not -1 and
        their positions < `actions_count_cpu` (i.e., rows
        `post_bwd_evicted_indices_cpu[:actions_count_cpu] != -1` in
        post_bwd_evicted_indices_cpu) will be evicted.

        Args:
            grad (Tensor): Unused gradient tensor

        Returns:
            None
        """
        with record_function("## ssd_evict_from_scratch_pad_pipeline ##"):
            current_stream = torch.cuda.current_stream()
            current_stream.record_event(self.ssd_event_backward)

            assert (
                len(self.ssd_scratch_pad_eviction_data) > 0
            ), "There must be at least one scratch pad"

            (
                inserted_rows,
                post_bwd_evicted_indices_cpu,
                actions_count_cpu,
                do_evict,
            ) = self.ssd_scratch_pad_eviction_data.pop(0)

            if not do_evict:
                return

            self.evict(
                rows=inserted_rows,
                indices_cpu=post_bwd_evicted_indices_cpu,
                actions_count_cpu=actions_count_cpu,
                stream=self.ssd_eviction_stream,
                pre_event=self.ssd_event_backward,
                post_event=self.ssd_event_sp_evict,
                is_rows_uvm=True,
                name="scratch_pad",
            )

            if self.prefetch_stream:
                self.prefetch_stream.wait_stream(current_stream)

    def _update_cache_counter_and_pointers(
        self,
        module: nn.Module,
        grad_input: Union[Tuple[Tensor, ...], Tensor],
    ) -> None:
        """
        Update cache line locking counter and pointers before backward
        TBE. This is a hook that is called before the backward of TBE

        Update cache line counter:

        We ensure that cache prefetching does not execute concurrently
        with the backward TBE. Therefore, it is safe to unlock the
        cache lines used in current iteration before backward TBE.

        Update pointers:

        Now some rows that are used in both the current iteration and
        the next iteration are moved (1) from the current iteration's
        scratch pad into the next iteration's scratch pad or (2) from
        the current iteration's scratch pad into the L1 cache

        To ensure that the TBE backward kernel accesses valid data,
        here we update the pointers of these rows in the current
        iteration's `lxu_cache_ptrs` to point to either L1 cache or
        the next iteration scratch pad

        Args:
            module (nn.Module): Unused
            grad_input (Union[Tuple[Tensor, ...], Tensor]): Unused

        Returns:
            None
        """
        if self.prefetch_stream:
            # Ensure that prefetch is done
            torch.cuda.current_stream().wait_stream(self.prefetch_stream)

        assert self.current_iter_data is not None, "current_iter_data must be set"

        curr_data: IterData = self.current_iter_data

        if curr_data.lxu_cache_locations.numel() == 0:
            return

        with record_function("## ssd_update_cache_counter_and_pointers ##"):
            # Unlock the cache lines
            torch.ops.fbgemm.lxu_cache_locking_counter_decrement(
                self.lxu_cache_locking_counter,
                curr_data.lxu_cache_locations,
            )

            # Recompute linear_cache_indices to save memory
            linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
                self.hash_size_cumsum,
                curr_data.indices,
                curr_data.offsets,
                curr_data.B_offsets,
                curr_data.max_B,
            )
            (
                linear_unique_indices,
                linear_unique_indices_length,
                unique_indices_count,
                linear_index_inverse_indices,
            ) = get_unique_indices_v2(
                linear_cache_indices,
                self.total_hash_size,
                compute_count=True,
                compute_inverse_indices=True,
            )
            unique_indices_count_cumsum = torch.ops.fbgemm.asynchronous_complete_cumsum(
                unique_indices_count
            )

            # Look up the cache to check which indices in the scratch
            # pad are moved to L1
            torch.ops.fbgemm.lxu_cache_lookup(
                linear_cache_indices,
                self.lxu_cache_state,
                self.total_hash_size,
                gather_cache_stats=False,  # not collecting cache stats
                lxu_cache_locations_output=curr_data.lxu_cache_locations,
            )

            if len(self.ssd_location_update_data) == 0:
                return

            (sp_curr_next_map, inserted_rows_next) = self.ssd_location_update_data.pop(
                0
            )

            # Update poitners
            torch.ops.fbgemm.ssd_update_row_addrs(
                ssd_row_addrs_curr=curr_data.lxu_cache_ptrs,
                inserted_ssd_weights_curr_next_map=sp_curr_next_map,
                lxu_cache_locations_curr=curr_data.lxu_cache_locations,
                linear_index_inverse_indices_curr=linear_index_inverse_indices,
                unique_indices_count_cumsum_curr=unique_indices_count_cumsum,
                cache_set_inverse_indices_curr=curr_data.cache_set_inverse_indices,
                lxu_cache_weights=self.lxu_cache_weights,
                inserted_ssd_weights_next=inserted_rows_next,
                unique_indices_length_curr=curr_data.actions_count_gpu,
            )

    def prefetch(
        self,
        indices: Tensor,
        offsets: Tensor,
        forward_stream: Optional[torch.cuda.Stream] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
    ) -> None:
        if self.prefetch_stream is None and forward_stream is not None:
            # Set the prefetch stream to the current stream
            self.prefetch_stream = torch.cuda.current_stream()
            assert (
                self.prefetch_stream != forward_stream
            ), "prefetch_stream and forward_stream should not be the same stream"

            current_stream = torch.cuda.current_stream()
            # Record tensors on the current stream
            indices.record_stream(current_stream)
            offsets.record_stream(current_stream)

        indices, offsets, _, vbe_metadata = self.prepare_inputs(
            indices,
            offsets,
            per_sample_weights=None,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

        self._prefetch(
            indices,
            offsets,
            vbe_metadata,
            forward_stream,
        )

    def _prefetch(  # noqa C901
        self,
        indices: Tensor,
        offsets: Tensor,
        vbe_metadata: Optional[invokers.lookup_args.VBEMetadata] = None,
        forward_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        # TODO: Refactor prefetch
        current_stream = torch.cuda.current_stream()

        B_offsets = None
        max_B = -1
        if vbe_metadata is not None:
            B_offsets = vbe_metadata.B_offsets
            max_B = vbe_metadata.max_B

        with record_function("## ssd_prefetch {} ##".format(self.timestep)):
            if self.gather_ssd_cache_stats:
                self.local_ssd_cache_stats.zero_()

            # Linearize indices
            linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
                self.hash_size_cumsum,
                indices,
                offsets,
                B_offsets,
                max_B,
            )

            self.timestep += 1
            self.timesteps_prefetched.append(self.timestep)

            # Lookup and virtually insert indices into L1. After this operator,
            # we know:
            # (1) which cache lines can be evicted
            # (2) which rows are already in cache (hit)
            # (3) which rows are missed and can be inserted later (missed, but
            #     not conflict missed)
            # (4) which rows are missed but CANNOT be inserted later (conflict
            #     missed)
            (
                inserted_indices,
                evicted_indices,
                assigned_cache_slots,
                actions_count_gpu,
                linear_index_inverse_indices,
                unique_indices_count_cumsum,
                cache_set_inverse_indices,
                unique_indices_length,
            ) = torch.ops.fbgemm.ssd_cache_populate_actions(
                linear_cache_indices,
                self.total_hash_size,
                self.lxu_cache_state,
                self.timestep,
                1,  # for now assume prefetch_dist == 1
                self.lru_state,
                self.gather_ssd_cache_stats,
                self.local_ssd_cache_stats,
                lock_cache_line=self.prefetch_pipeline,
                lxu_cache_locking_counter=self.lxu_cache_locking_counter,
            )

            # Compute cache locations (rows that are hit are missed but can be
            # inserted will have cache locations != -1)
            with record_function("## ssd_tbe_lxu_cache_lookup ##"):
                lxu_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
                    linear_cache_indices,
                    self.lxu_cache_state,
                    self.total_hash_size,
                    self.gather_ssd_cache_stats,
                    self.local_ssd_cache_stats,
                )

            # Defrag indices based on evicted_indices (removing -1 and making
            # the non -1 elements contiguous). We need to do this because the
            # number of rows in `lxu_cache_evicted_weights` might be smaller
            # than the number of elements in `evicted_indices`. Without this
            # step, we can run into the index out of bound issue
            current_stream.wait_event(self.ssd_event_cache_evict)
            torch.ops.fbgemm.compact_indices(
                compact_indices=[
                    self.lxu_cache_evicted_indices,
                    self.lxu_cache_evicted_slots,
                ],
                compact_count=self.lxu_cache_evicted_count,
                indices=[evicted_indices, assigned_cache_slots],
                masks=torch.where(evicted_indices != -1, 1, 0),
                count=actions_count_gpu,
            )

            with record_function("## ssd_d2h_inserted_indices ##"):
                # Transfer actions_count and insert_indices right away to
                # incrase an overlap opportunity
                actions_count_cpu, inserted_indices_cpu = (
                    self.to_pinned_cpu_on_stream_wait_on_another_stream(
                        tensors=[
                            actions_count_gpu,
                            inserted_indices,
                        ],
                        stream=self.ssd_memcpy_stream,
                        stream_to_wait_on=current_stream,
                        post_event=self.ssd_event_get_inputs_cpy,
                    )
                )

            # Copy rows to be evicted into a separate buffer (will be evicted
            # later in the prefetch step)
            with record_function("## ssd_compute_evicted_rows ##"):
                torch.ops.fbgemm.masked_index_select(
                    self.lxu_cache_evicted_weights,
                    self.lxu_cache_evicted_slots,
                    self.lxu_cache_weights,
                    self.lxu_cache_evicted_count,
                )

            # Allocation a scratch pad for the current iteration. The scratch
            # pad is a UVA tensor
            inserted_rows_shape = (assigned_cache_slots.numel(), self.max_D)
            if linear_cache_indices.numel() > 0:
                inserted_rows = torch.ops.fbgemm.new_unified_tensor(
                    torch.zeros(
                        1,
                        device=self.current_device,
                        dtype=self.lxu_cache_weights.dtype,
                    ),
                    inserted_rows_shape,
                    is_host_mapped=self.uvm_host_mapped,
                )
            else:
                inserted_rows = torch.empty(
                    inserted_rows_shape,
                    dtype=self.lxu_cache_weights.dtype,
                    device=self.current_device,
                )

            if self.prefetch_pipeline and len(self.ssd_scratch_pads) > 0:
                # Look up all missed indices from the previous iteration's
                # scratch pad (do this only if pipeline prefetching is being
                # used)
                with record_function("## ssd_lookup_scratch_pad ##"):
                    # Get the previous scratch pad
                    (
                        inserted_rows_prev,
                        post_bwd_evicted_indices_cpu_prev,
                        actions_count_cpu_prev,
                    ) = self.ssd_scratch_pads.pop(0)

                    # Inserted indices that are found in the scratch pad
                    # from the previous iteration
                    sp_prev_curr_map_cpu = torch.empty(
                        inserted_indices_cpu.shape,
                        dtype=inserted_indices_cpu.dtype,
                        pin_memory=True,
                    )

                    # Conflict missed indices from the previous iteration that
                    # overlap with the current iterations's inserted indices
                    sp_curr_prev_map_cpu = torch.empty(
                        post_bwd_evicted_indices_cpu_prev.shape,
                        dtype=torch.int,
                        pin_memory=True,
                    ).fill_(-1)

                    # Ensure that the necessary D2H transfers are done
                    current_stream.wait_event(self.ssd_event_get_inputs_cpy)
                    # Ensure that the previous iteration's scratch pad indices
                    # insertion is complete
                    current_stream.wait_event(self.ssd_event_sp_idxq_insert)

                    # Before entering this function: inserted_indices_cpu
                    # contains all linear indices that are missed from the
                    # L1 cache
                    #
                    # After this function: inserted indices that are found
                    # in the scratch pad from the previous iteration are
                    # stored in sp_prev_curr_map_cpu, while the rests are
                    # stored in inserted_indices_cpu
                    #
                    # An invalid index is -1 or its position >
                    # actions_count_cpu
                    self.record_function_via_dummy_profile(
                        "## ssd_lookup_mask_and_pop_front ##",
                        self.scratch_pad_idx_queue.lookup_mask_and_pop_front_cuda,
                        sp_prev_curr_map_cpu,  # scratch_pad_prev_curr_map
                        sp_curr_prev_map_cpu,  # scratch_pad_curr_prev_map
                        post_bwd_evicted_indices_cpu_prev,  # scratch_pad_indices_prev
                        inserted_indices_cpu,  # inserted_indices_curr
                        actions_count_cpu,  # count_curr
                    )

                    # Mark scratch pad index queue lookup completion
                    current_stream.record_event(self.ssd_event_sp_idxq_lookup)

                    # Transfer sp_prev_curr_map_cpu to GPU
                    sp_prev_curr_map_gpu = sp_prev_curr_map_cpu.cuda(non_blocking=True)
                    # Transfer sp_curr_prev_map_cpu to GPU
                    sp_curr_prev_map_gpu = sp_curr_prev_map_cpu.cuda(non_blocking=True)

                    # Previously actions_count_gpu was recorded on another
                    # stream. Thus, we need to record it on this stream
                    actions_count_gpu.record_stream(current_stream)

                    # Copy data from the previous iteration's scratch pad to
                    # the current iteration's scratch pad
                    torch.ops.fbgemm.masked_index_select(
                        inserted_rows,
                        sp_prev_curr_map_gpu,
                        inserted_rows_prev,
                        actions_count_gpu,
                        use_pipeline=self.prefetch_pipeline,
                    )

                    # Record the tensors that will be pushed into a queue
                    # on the forward stream
                    if forward_stream:
                        sp_curr_prev_map_gpu.record_stream(forward_stream)

                    # Store info for evicting the previous iteration's
                    # scratch pad after the corresponding backward pass is
                    # done
                    self.ssd_location_update_data.append(
                        (
                            sp_curr_prev_map_gpu,
                            inserted_rows,
                        )
                    )

            # Ensure the previous iterations eviction is complete
            current_stream.wait_event(self.ssd_event_sp_evict)
            # Ensure that D2H is done
            current_stream.wait_event(self.ssd_event_get_inputs_cpy)

            if self.gather_ssd_cache_stats:
                # call to collect past SSD IO dur right before next rocksdb IO

                self.ssd_cache_stats = torch.add(
                    self.ssd_cache_stats, self.local_ssd_cache_stats
                )
                self._report_ssd_stats()

            # Fetch data from SSD
            if linear_cache_indices.numel() > 0:
                self.record_function_via_dummy_profile(
                    "## ssd_get ##",
                    self.ssd_db.get_cuda,
                    inserted_indices_cpu,
                    inserted_rows,
                    actions_count_cpu,
                )

            # Record an event to mark the completion of `get_cuda`
            current_stream.record_event(self.ssd_event_get)

            # Copy rows from the current iteration's scratch pad to L1
            torch.ops.fbgemm.masked_index_put(
                self.lxu_cache_weights,
                assigned_cache_slots,
                inserted_rows,
                actions_count_gpu,
                use_pipeline=self.prefetch_pipeline,
            )

            if linear_cache_indices.numel() > 0:
                # Evict rows from cache to SSD
                self.evict(
                    rows=self.lxu_cache_evicted_weights,
                    indices_cpu=self.lxu_cache_evicted_indices,
                    actions_count_cpu=self.lxu_cache_evicted_count,
                    stream=self.ssd_eviction_stream,
                    pre_event=self.ssd_event_get,
                    # Record completion event after scratch pad eviction
                    # instead since that happens after L1 eviction
                    post_event=self.ssd_event_cache_evict,
                    is_rows_uvm=True,
                    name="cache",
                    is_bwd=False,
                )

            # Generate row addresses (pointing to either L1 or the current
            # iteration's scratch pad)
            with record_function("## ssd_generate_row_addrs ##"):
                lxu_cache_ptrs, post_bwd_evicted_indices = (
                    torch.ops.fbgemm.ssd_generate_row_addrs(
                        lxu_cache_locations,
                        assigned_cache_slots,
                        linear_index_inverse_indices,
                        unique_indices_count_cumsum,
                        cache_set_inverse_indices,
                        self.lxu_cache_weights,
                        inserted_rows,
                        unique_indices_length,
                        inserted_indices,
                    )
                )

            with record_function("## ssd_d2h_post_bwd_evicted_indices ##"):
                # Transfer post_bwd_evicted_indices from GPU to CPU right away to
                # increase a chance of overlapping with compute in the default stream
                (post_bwd_evicted_indices_cpu,) = (
                    self.to_pinned_cpu_on_stream_wait_on_another_stream(
                        tensors=[post_bwd_evicted_indices],
                        stream=self.ssd_eviction_stream,
                        stream_to_wait_on=current_stream,
                        post_event=None,
                    )
                )

            if self.prefetch_pipeline:
                # Insert the current iteration's conflict miss indices in the index
                # queue for future lookup.
                #
                # post_bwd_evicted_indices_cpu is transferred on the
                # ssd_eviction_stream stream so it does not need stream
                # synchronization
                #
                # actions_count_cpu is transferred on the ssd_memcpy_stream stream.
                # Thus, we have to explicitly sync the stream
                with torch.cuda.stream(self.ssd_eviction_stream):
                    # Ensure that actions_count_cpu transfer is done
                    self.ssd_eviction_stream.wait_event(self.ssd_event_get_inputs_cpy)
                    # Ensure that the scratch pad index queue look up is complete
                    self.ssd_eviction_stream.wait_event(self.ssd_event_sp_idxq_lookup)
                    self.record_function_via_dummy_profile(
                        "## ssd_scratch_pad_idx_queue_insert ##",
                        self.scratch_pad_idx_queue.insert_cuda,
                        post_bwd_evicted_indices_cpu,
                        actions_count_cpu,
                    )
                    # Mark the completion of scratch pad index insertion
                    self.ssd_eviction_stream.record_event(self.ssd_event_sp_idxq_insert)

            prefetch_data = (
                lxu_cache_ptrs,
                inserted_rows,
                post_bwd_evicted_indices_cpu,
                actions_count_cpu,
                actions_count_gpu,
                lxu_cache_locations,
                cache_set_inverse_indices,
            )

            # Record tensors on the forward stream
            if forward_stream is not None:
                for t in prefetch_data:
                    if t.is_cuda:
                        t.record_stream(forward_stream)

            if self.prefetch_pipeline:
                # Store scratch pad info for the lookup in the next iteration
                # prefetch
                self.ssd_scratch_pads.append(
                    (
                        inserted_rows,
                        post_bwd_evicted_indices_cpu,
                        actions_count_cpu,
                    )
                )

            # Store scratch pad info for post backward eviction
            self.ssd_scratch_pad_eviction_data.append(
                (
                    inserted_rows,
                    post_bwd_evicted_indices_cpu,
                    actions_count_cpu,
                    linear_cache_indices.numel() > 0,
                )
            )

            # Store data for forward
            self.ssd_prefetch_data.append(prefetch_data)

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
            ), (
                "Variable batch size TBE support is enabled for "
                "OptimType.EXACT_ROWWISE_ADAGRAD and "
                "ENSEMBLE_ROWWISE_ADAGRAD only"
            )
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
        # pyre-fixme[7]: Expected `Tensor` but got implicit return value of `None`.
    ) -> Tensor:
        indices, offsets, per_sample_weights, vbe_metadata = self.prepare_inputs(
            indices, offsets, per_sample_weights, batch_size_per_feature_per_rank
        )

        if len(self.timesteps_prefetched) == 0:

            with self._recording_to_timer(
                self.ssd_prefetch_read_timer,
                context=self.step,
                stream=torch.cuda.current_stream(),
            ), self._recording_to_timer(
                self.ssd_prefetch_evict_timer,
                context=self.step,
                stream=self.ssd_eviction_stream,
            ):
                self._prefetch(indices, offsets, vbe_metadata)

        assert len(self.ssd_prefetch_data) > 0

        (
            lxu_cache_ptrs,
            inserted_rows,
            post_bwd_evicted_indices_cpu,
            actions_count_cpu,
            actions_count_gpu,
            lxu_cache_locations,
            cache_set_inverse_indices,
        ) = self.ssd_prefetch_data.pop(0)

        # Storing current iteration data for future use
        self.current_iter_data = IterData(
            indices,
            offsets,
            lxu_cache_locations,
            lxu_cache_ptrs,
            actions_count_gpu,
            cache_set_inverse_indices,
            vbe_metadata.B_offsets,
            vbe_metadata.max_B,
        )

        common_args = invokers.lookup_args_ssd.CommonArgs(
            placeholder_autograd_tensor=self.placeholder_autograd_tensor,
            output_dtype=self.output_dtype,
            dev_weights=self.weights_dev,
            host_weights=self.weights_host,
            uvm_weights=self.weights_uvm,
            lxu_cache_weights=self.lxu_cache_weights,
            weights_placements=self.weights_placements,
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
            lxu_cache_locations=lxu_cache_ptrs,
            uvm_cache_stats=None,
            # Unused arguments
            is_experimental=False,
            use_uniq_cache_locations_bwd=False,
            use_homogeneous_placements=True,
            # The keys for ssd_tensors are controlled by ssd_tensors in
            # codegen/genscript/optimizer_args.py
            ssd_tensors={
                "row_addrs": lxu_cache_ptrs,
                "inserted_rows": inserted_rows,
                "post_bwd_evicted_indices": post_bwd_evicted_indices_cpu,
                "actions_count": actions_count_cpu,
            },
            # pyre-fixme[6]: Expected `lookup_args_ssd.VBEMetadata` but got `lookup_args.VBEMetadata`
            vbe_metadata=vbe_metadata,
        )

        self.timesteps_prefetched.pop(0)
        self.step += 1

        if self.optimizer == OptimType.EXACT_SGD:
            raise AssertionError(
                "SSDTableBatchedEmbeddingBags currently does not support SGD"
            )
            return invokers.lookup_sgd_ssd.invoke(common_args, self.optimizer_args)

        momentum1 = invokers.lookup_args_ssd.Momentum(
            dev=self.momentum1_dev,
            host=self.momentum1_host,
            uvm=self.momentum1_uvm,
            offsets=self.momentum1_offsets,
            placements=self.momentum1_placements,
        )

        if self.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
            # pyre-fixme[7]: Expected `Tensor` but got implicit return value of `None`.
            return invokers.lookup_rowwise_adagrad_ssd.invoke(
                common_args, self.optimizer_args, momentum1
            )

    @torch.jit.ignore
    def debug_split_optimizer_states(self) -> List[Tuple[torch.Tensor]]:
        """
        Returns a list of states, split by table
        Testing only
        """
        (rows, _) = zip(*self.embedding_specs)

        rows_cumsum = [0] + list(itertools.accumulate(rows))

        return [
            (
                self.momentum1_dev.detach()[rows_cumsum[t] : rows_cumsum[t + 1]].view(
                    row
                ),
            )
            for t, row in enumerate(rows)
        ]

    @torch.jit.export
    def debug_split_embedding_weights(self) -> List[torch.Tensor]:
        """
        Returns a list of weights, split by table.

        Testing only, very slow.
        """
        (rows, _) = zip(*self.embedding_specs)

        rows_cumsum = [0] + list(itertools.accumulate(rows))
        splits = []
        get_event = torch.cuda.Event()

        for t, (row, dim) in enumerate(self.embedding_specs):
            weights = torch.empty(
                (row, self.max_D), dtype=self.weights_precision.as_dtype()
            )
            self.ssd_db.get_cuda(
                torch.arange(rows_cumsum[t], rows_cumsum[t + 1]).to(torch.int64),
                weights,
                torch.as_tensor([row]),
            )
            splits.append(weights)

        # Record the event to create a dependency between get_cuda's callback
        # function and the kernel on the GPU default stream (the intention is
        # actually to synchronize between the callback CPU thread and the
        # Python CPU thread but we do not have a mechanism to explicitly sync
        # between them)
        get_event.record()

        # Synchronize to make sure that the callback function in get_cuda
        # completes (here the CPU thread is blocked until get_event is done)
        get_event.synchronize()

        # Reshape the weight tensors (this can be expensive, however, this
        # function is for debugging only)
        for t, (row, dim) in enumerate(self.embedding_specs):
            weight = splits[t]
            weight = weight[:, :dim].contiguous()
            assert weight.shape == (row, dim), "Shapes mismatch"
            splits[t] = weight

        return splits

    @torch.jit.export
    def split_embedding_weights(
        self,
        no_snapshot: bool = True,
        should_flush: bool = False,
    ) -> List[PartiallyMaterializedTensor]:
        """
        This method is intended to be used by the checkpointing engine
        only.

        Since we cannot materialize SSD backed tensors fully in CPU memory,
        we would create a KVTensorWrapper (ssd_split_table_batched_embeddings.cpp)
        for each table (shard), which allows caller to read the weights
        using `narrow()` in a rolling-window manner.
        Args:
            should_flush (bool): Flush caches if True. Note: this is an expensive
                                 operation, only set to True when necessary.

        Returns:
            a list of partially materialized tensors, each representing a table
        """
        # Force device synchronize for now
        torch.cuda.synchronize()
        # Create a snapshot
        if no_snapshot:
            snapshot_handle = None
        else:
            if should_flush:
                # Flush L1 and L2 caches
                self.flush()
            snapshot_handle = self.ssd_db.create_snapshot()

        dtype = self.weights_precision.as_dtype()
        splits = []

        row_offset = 0
        for emb_height, emb_dim in self.embedding_specs:
            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                db=self.ssd_db,
                shape=[emb_height, emb_dim],
                dtype=dtype,
                row_offset=row_offset,
                snapshot_handle=snapshot_handle,
            )
            row_offset += emb_height
            splits.append(PartiallyMaterializedTensor(tensor_wrapper))
        return splits

    @torch.jit.export
    def set_learning_rate(self, lr: float) -> None:
        """
        Sets the learning rate.
        """
        self._set_learning_rate(lr)

    def get_learning_rate(self) -> float:
        """
        Sets the learning rate.

        Args:
            lr (float): The learning rate value to set to
        """
        return self.optimizer_args.learning_rate

    @torch.jit.ignore
    def _set_learning_rate(self, lr: float) -> float:
        """
        Helper function to script `set_learning_rate`.
        Note that returning None does not work.
        """
        self.optimizer_args = self.optimizer_args._replace(learning_rate=lr)
        return 0.0

    def flush(self) -> None:
        active_slots_mask = self.lxu_cache_state != -1

        active_weights_gpu = self.lxu_cache_weights[active_slots_mask.view(-1)].view(
            -1, self.max_D
        )
        active_ids_gpu = self.lxu_cache_state.view(-1)[active_slots_mask.view(-1)]

        active_weights_cpu = active_weights_gpu.cpu()
        active_ids_cpu = active_ids_gpu.cpu()

        torch.cuda.current_stream().wait_stream(self.ssd_eviction_stream)

        torch.cuda.synchronize()
        self.ssd_db.set(
            active_ids_cpu,
            active_weights_cpu,
            torch.tensor([active_ids_cpu.numel()]),
        )
        self.ssd_db.flush()

    def prepare_inputs(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], invokers.lookup_args.VBEMetadata]:
        """
        Prepare TBE inputs
        """
        # Generate VBE metadata
        vbe_metadata = self._generate_vbe_metadata(
            offsets, batch_size_per_feature_per_rank
        )

        # Force casting indices and offsets to long
        (indices, offsets) = indices.long(), offsets.long()

        # Force casting per_sample_weights to float
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.float()

        if self.bounds_check_mode_int != BoundsCheckMode.NONE.value:
            torch.ops.fbgemm.bounds_check_indices(
                self.rows_per_table,
                indices,
                offsets,
                self.bounds_check_mode_int,
                self.bounds_check_warning,
                per_sample_weights,
                B_offsets=vbe_metadata.B_offsets,
                max_B=vbe_metadata.max_B,
            )

        return indices, offsets, per_sample_weights, vbe_metadata

    @torch.jit.ignore
    def _report_ssd_stats(self) -> None:
        """
        All ssd stats report function entrance
        """
        if self.stats_reporter is None:
            return

        if not self.stats_reporter.should_report(self.step):
            return
        self._report_ssd_l1_cache_stats()
        self._report_ssd_io_stats()
        self._report_ssd_mem_usage()
        self._report_l2_cache_perf_stats()

    @torch.jit.ignore
    def _report_ssd_l1_cache_stats(self) -> None:
        """
        Each iteration we will record cache stats about L1 SSD cache in ssd_cache_stats tensor
        this function extract those stats and report it with stats_reporter
        """
        passed_steps = self.step - self.last_reported_step
        if passed_steps == 0:
            return

        # ssd hbm cache stats

        ssd_cache_stats = self.ssd_cache_stats.tolist()
        if len(self.last_reported_ssd_stats) == 0:
            self.last_reported_ssd_stats = [0.0] * len(ssd_cache_stats)
        ssd_cache_stats_delta: List[float] = [0.0] * len(ssd_cache_stats)
        for i in range(len(ssd_cache_stats)):
            ssd_cache_stats_delta[i] = (
                ssd_cache_stats[i] - self.last_reported_ssd_stats[i]
            )
        self.last_reported_step = self.step
        self.last_reported_ssd_stats = ssd_cache_stats
        element_size = self.lxu_cache_weights.element_size()

        for stat_index in UVMCacheStatsIndex:
            # pyre-ignore
            self.stats_reporter.report_data_amount(
                iteration_step=self.step,
                event_name=f"ssd_tbe.prefetch.cache_stats_by_data_size.{stat_index.name.lower()}",
                data_bytes=int(
                    ssd_cache_stats_delta[stat_index.value]
                    * element_size
                    * self.max_D
                    / passed_steps
                ),
            )
            # pyre-ignore
            self.stats_reporter.report_data_amount(
                iteration_step=self.step,
                event_name=f"ssd_tbe.prefetch.cache_stats.{stat_index.name.lower()}",
                data_bytes=int(ssd_cache_stats_delta[stat_index.value] / passed_steps),
            )

    @torch.jit.ignore
    def _report_ssd_io_stats(self) -> None:
        """
        EmbeddingRocksDB will hold stats for total read/write duration in fwd/bwd
        this function fetch the stats from EmbeddingRocksDB and report it with stats_reporter
        """
        ssd_io_duration = self.ssd_db.get_rocksdb_io_duration(
            self.step, self.stats_reporter.report_interval  # pyre-ignore
        )

        if len(ssd_io_duration) != 5:
            logging.error("ssd io duration should have 5 elements")
            return

        ssd_read_dur_us = ssd_io_duration[0]
        fwd_rocksdb_read_dur = ssd_io_duration[1]
        fwd_l1_eviction_dur = ssd_io_duration[2]
        bwd_l1_cnflct_miss_write_back_dur = ssd_io_duration[3]
        flush_write_dur = ssd_io_duration[4]

        # pyre-ignore
        self.stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="ssd.io_duration.read_us",
            duration_ms=ssd_read_dur_us,
            time_unit="us",
        )
        # pyre-ignore
        self.stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="ssd.io_duration.write.fwd_rocksdb_read_us",
            duration_ms=fwd_rocksdb_read_dur,
            time_unit="us",
        )
        # pyre-ignore
        self.stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="ssd.io_duration.write.fwd_l1_eviction_us",
            duration_ms=fwd_l1_eviction_dur,
            time_unit="us",
        )
        # pyre-ignore
        self.stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="ssd.io_duration.write.bwd_l1_cnflct_miss_write_back_us",
            duration_ms=bwd_l1_cnflct_miss_write_back_dur,
            time_unit="us",
        )
        # pyre-ignore
        self.stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="ssd.io_duration.write.flush_write_us",
            duration_ms=flush_write_dur,
            time_unit="us",
        )

    @torch.jit.ignore
    def _report_ssd_mem_usage(
        self,
    ) -> None:
        """
        rocskdb has internal stats for dram mem usage, here we call EmbeddingRocksDB to
        extract those stats out and report it with stats_reporter
        """
        mem_usage_list = self.ssd_db.get_mem_usage()
        block_cache_usage = mem_usage_list[0]
        estimate_table_reader_usage = mem_usage_list[1]
        memtable_usage = mem_usage_list[2]
        block_cache_pinned_usage = mem_usage_list[3]

        # pyre-ignore
        self.stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name="ssd.mem_usage.block_cache",
            data_bytes=block_cache_usage,
        )
        # pyre-ignore
        self.stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name="ssd.mem_usage.estimate_table_reader",
            data_bytes=estimate_table_reader_usage,
        )
        # pyre-ignore
        self.stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name="ssd.mem_usage.memtable",
            data_bytes=memtable_usage,
        )
        # pyre-ignore
        self.stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name="ssd.mem_usage.block_cache_pinned",
            data_bytes=block_cache_pinned_usage,
        )

    @torch.jit.ignore
    def _report_l2_cache_perf_stats(self) -> None:
        """
        EmbeddingKVDB will hold stats for L2+SSD performance in fwd/bwd
        this function fetch the stats from EmbeddingKVDB and report it with stats_reporter
        """
        if self.stats_reporter is None:
            return

        stats_reporter: TBEStatsReporter = self.stats_reporter
        if not stats_reporter.should_report(self.step):
            return

        l2_cache_perf_stats = self.ssd_db.get_l2cache_perf(
            self.step, stats_reporter.report_interval  # pyre-ignore
        )

        if len(l2_cache_perf_stats) != 15:
            logging.error("l2 perf stats should have 15 elements")
            return

        num_cache_misses = l2_cache_perf_stats[0]
        num_lookups = l2_cache_perf_stats[1]
        get_total_duration = l2_cache_perf_stats[2]
        get_cache_lookup_total_duration = l2_cache_perf_stats[3]
        get_cache_lookup_wait_filling_thread_duration = l2_cache_perf_stats[4]
        get_weights_fillup_total_duration = l2_cache_perf_stats[5]
        get_cache_memcpy_duration = l2_cache_perf_stats[6]
        total_cache_update_duration = l2_cache_perf_stats[7]
        get_tensor_copy_for_cache_update_duration = l2_cache_perf_stats[8]
        set_tensor_copy_for_cache_update_duration = l2_cache_perf_stats[9]
        num_l2_evictions = l2_cache_perf_stats[10]

        l2_cache_free_bytes = l2_cache_perf_stats[11]
        l2_cache_capacity = l2_cache_perf_stats[12]

        set_cache_lock_wait_duration = l2_cache_perf_stats[13]
        get_cache_lock_wait_duration = l2_cache_perf_stats[14]

        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name=self.l2_num_cache_misses_stats_name,
            data_bytes=num_cache_misses,
        )
        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name=self.l2_num_cache_lookups_stats_name,
            data_bytes=num_lookups,
        )
        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name=self.l2_num_cache_evictions_stats_name,
            data_bytes=num_l2_evictions,
        )
        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name=self.l2_cache_capacity_stats_name,
            data_bytes=l2_cache_capacity,
        )
        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name=self.l2_cache_free_mem_stats_name,
            data_bytes=l2_cache_free_bytes,
        )

        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="l2_cache.perf.get.total_duration_us",
            duration_ms=get_total_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="l2_cache.perf.get.cache_lookup_duration_us",
            duration_ms=get_cache_lookup_total_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="l2_cache.perf.get.cache_lookup_wait_filling_thread_duration_us",
            duration_ms=get_cache_lookup_wait_filling_thread_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="l2_cache.perf.get.weights_fillup_duration_us",
            duration_ms=get_weights_fillup_total_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="l2_cache.perf.get.cache_memcpy_duration_us",
            duration_ms=get_cache_memcpy_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="l2_cache.perf.total.cache_update_duration_us",
            duration_ms=total_cache_update_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="l2_cache.perf.get.tensor_copy_for_cache_update_duration_us",
            duration_ms=get_tensor_copy_for_cache_update_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="l2_cache.perf.set.tensor_copy_for_cache_update_duration_us",
            duration_ms=set_tensor_copy_for_cache_update_duration,
            time_unit="us",
        )

        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="l2_cache.perf.get.cache_lock_wait_duration_us",
            duration_ms=get_cache_lock_wait_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="l2_cache.perf.set.cache_lock_wait_duration_us",
            duration_ms=set_cache_lock_wait_duration,
            time_unit="us",
        )

    # pyre-ignore
    def _recording_to_timer(
        self, timer: Optional[AsyncSeriesTimer], **kwargs: Any
    ) -> Any:
        """
        helper function to call AsyncSeriesTimer, wrap it inside the kernels we want to record
        """
        if self.stats_reporter is not None and self.stats_reporter.should_report(
            self.step
        ):
            assert (
                timer
            ), "We shouldn't be here, async timer must have been initiated if reporter is present."
            return timer.recording(**kwargs)
        # No-Op context manager
        return contextlib.nullcontext()
