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
import math
import os
import tempfile
import threading
import time
from functools import cached_property
from math import floor, log2
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
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
    BackendType,
    BoundsCheckMode,
    CacheAlgorithm,
    EmbeddingLocation,
    get_bounds_check_version_for_platform,
    KVZCHParams,
    PoolingMode,
    SplitState,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    apply_split_helper,
    CounterBasedRegularizationDefinition,
    CowClipDefinition,
    RESParams,
    UVMCacheStatsIndex,
    WeightDecayMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import (
    generate_vbe_metadata,
    is_torchdynamo_compiling,
)
from torch import distributed as dist, nn, Tensor  # usort:skip
from dataclasses import dataclass

from torch.autograd.profiler import record_function

from ..cache import get_unique_indices_v2
from .common import ASSOC, pad4, tensor_pad4
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


@dataclass
class KVZCHCachedData:
    cached_optimizer_states_per_table: List[List[torch.Tensor]]
    cached_weight_tensor_per_table: List[torch.Tensor]
    cached_id_tensor_per_table: List[torch.Tensor]
    cached_bucket_splits: List[torch.Tensor]


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
    res_params: RESParams
    table_names: List[str]

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
        backend_type: BackendType = BackendType.SSD,
        kv_zch_params: Optional[KVZCHParams] = None,
        enable_raw_embedding_streaming: bool = False,  # whether enable raw embedding streaming
        res_params: Optional[RESParams] = None,  # raw embedding streaming sharding info
        flushing_block_size: int = 2_000_000_000,  # 2GB
        table_names: Optional[List[str]] = None,
        optimizer_state_dtypes: Dict[str, SparseType] = {},  # noqa: B006
    ) -> None:
        super(SSDTableBatchedEmbeddingBags, self).__init__()

        # Set the optimizer
        assert optimizer in (
            OptimType.EXACT_ROWWISE_ADAGRAD,
            OptimType.PARTIAL_ROWWISE_ADAM,
        ), f"Optimizer {optimizer} is not supported by SSDTableBatchedEmbeddingBags"
        self.optimizer = optimizer

        # Set the table weight and output dtypes
        assert weights_precision in (SparseType.FP32, SparseType.FP16)
        self.weights_precision = weights_precision
        self.output_dtype: int = output_dtype.as_int()

        if self.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
            # Adagrad currently only supports FP32 for momentum1
            self.optimizer_state_dtypes: Dict[str, SparseType] = {
                "momentum1": SparseType.FP32,
            }
        else:
            self.optimizer_state_dtypes: Dict[str, SparseType] = optimizer_state_dtypes

        # Zero collision TBE configurations
        self.kv_zch_params = kv_zch_params
        self.backend_type = backend_type
        self.enable_optimizer_offloading: bool = False
        self.backend_return_whole_row: bool = False
        if self.kv_zch_params:
            self.kv_zch_params.validate()
            self.enable_optimizer_offloading = (
                # pyre-ignore [16]
                self.kv_zch_params.enable_optimizer_offloading
            )
            self.backend_return_whole_row = (
                # pyre-ignore [16]
                self.kv_zch_params.backend_return_whole_row
            )

            if self.enable_optimizer_offloading:
                logging.info("Optimizer state offloading is enabled")
            if self.backend_return_whole_row:
                assert (
                    self.backend_type == BackendType.DRAM
                ), f"Only DRAM backend supports backend_return_whole_row, but got {self.backend_type}"
                logging.info(
                    "Backend will return whole row including metaheader, weight and optimizer for checkpoint"
                )

        self.pooling_mode = pooling_mode
        self.bounds_check_mode_int: int = bounds_check_mode.value
        self.embedding_specs = embedding_specs
        self.table_names = table_names if table_names is not None else []
        (rows, dims) = zip(*embedding_specs)
        T_ = len(self.embedding_specs)
        assert T_ > 0
        # pyre-fixme[8]: Attribute has type `device`; used as `int`.
        self.current_device: torch.device = torch.cuda.current_device()

        self.enable_raw_embedding_streaming = enable_raw_embedding_streaming
        # initialize the raw embedding streaming related variables
        self.res_params: RESParams = res_params or RESParams()
        if self.enable_raw_embedding_streaming:
            self.res_params.table_sizes = [0] + list(itertools.accumulate(rows))
            res_port_from_env = os.getenv("LOCAL_RES_PORT")
            self.res_params.res_server_port = (
                int(res_port_from_env) if res_port_from_env else 0
            )
            logging.info(
                f"get env {self.res_params.res_server_port=}, at rank {dist.get_rank()}, with {self.res_params=}"
            )

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

        # Sum of row length of all tables
        self.total_D: int = D_offsets[-1]

        # Max number of elements required to store a row in the cache
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
        self.register_buffer(
            "table_hash_size_cumsum",
            torch.tensor(
                hash_size_cumsum, device=self.current_device, dtype=torch.int64
            ),
        )
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
        self.register_buffer(
            "table_dims",
            torch.tensor(dims, device="cpu", dtype=torch.int64),
        )

        (info_B_num_bits_, info_B_mask_) = torch.ops.fbgemm.get_infos_metadata(
            self.D_offsets,  # unused tensor
            1,  # max_B
            T,  # T
        )
        self.info_B_num_bits: int = info_B_num_bits_
        self.info_B_mask: int = info_B_mask_

        assert cache_sets > 0
        element_size = weights_precision.bit_rate() // 8
        assert (
            element_size == 4 or element_size == 2
        ), f"Invalid element size {element_size}"
        cache_size = cache_sets * ASSOC * element_size * self.cache_row_dim
        logging.info(
            f"Using cache for SSD with admission algorithm "
            f"{CacheAlgorithm.LRU}, {cache_sets} sets, stored on {'DEVICE' if ssd_cache_location is EmbeddingLocation.DEVICE else 'MANAGED'} with {ssd_rocksdb_shards} shards, "
            f"SSD storage directory: {ssd_storage_directory}, "
            f"Memtable Flush Period: {ssd_memtable_flush_period}, "
            f"Memtable Flush Offset: {ssd_memtable_flush_offset}, "
            f"Desired L0 files per compaction: {ssd_l0_files_per_compact}, "
            f"Cache size: {cache_size / 1024.0 / 1024.0 / 1024.0 : .2f}GB, "
            f"weights precision: {weights_precision}, "
            f"output dtype: {output_dtype}, "
            f"chunk size in bulk init: {bulk_init_chunk_size} bytes, backend_type: {backend_type}, "
            f"kv_zch_params: {kv_zch_params}, "
            f"embedding spec: {embedding_specs}"
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
        self.last_flush_step = -1

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
                    [cache_sets * ASSOC, self.cache_row_dim],
                    is_host_mapped=self.uvm_host_mapped,
                ),
            )
        else:
            self.register_buffer(
                "lxu_cache_weights",
                torch.zeros(
                    cache_sets * ASSOC,
                    self.cache_row_dim,
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

        # Store the iteration number on GPU and CPU (used for certain optimizers)
        persistent_iter_ = optimizer in (OptimType.PARTIAL_ROWWISE_ADAM,)
        self.register_buffer(
            "iter",
            torch.zeros(1, dtype=torch.int64, device=self.current_device),
            persistent=persistent_iter_,
        )
        self.iter_cpu: torch.Tensor = torch.zeros(1, dtype=torch.int64, device="cpu")

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

        self.record_function_via_dummy_profile: Callable[..., Any] = (
            self.record_function_via_dummy_profile_factory(use_dummy_profile)
        )

        os.makedirs(ssd_storage_directory, exist_ok=True)

        ssd_directory = tempfile.mkdtemp(
            prefix="ssd_table_batched_embeddings", dir=ssd_storage_directory
        )
        # logging.info("DEBUG: weights_precision {}".format(weights_precision))

        """
        ##################### for ZCH v.Next loading checkpoints Short Term Solution #######################
        weight_id tensor is the weight and optimizer keys, to load from checkpoint, weight_id tensor
        needs to be loaded first, then we can load the weight and optimizer tensors.
        However, the stateful checkpoint loading does not guarantee the tensor loading order, so we need
        to cache the weight_id, weight and optimizer tensors untils all data are loaded, then we can apply
        them to backend.
        Currently, we'll cache the weight_id, weight and optimizer tensors in the KVZCHCachedData class,
        and apply them to backend when all data are loaded. The downside of this solution is that we'll
        have to duplicate a whole tensor memory to backend before we can release the python tensor memory,
        which is not ideal.
        The longer term solution is to support the caching from the backend side, and allow streaming based
        data move from cached weight and optimizer to key/value format without duplicate one whole tensor's
        memory.
        """
        self._cached_kvzch_data: Optional[KVZCHCachedData] = None
        # initial embedding rows on this rank per table, this is used for loading checkpoint
        self.local_weight_counts: List[int] = [0] * T_
        # loading checkpoint flag, set by checkpoint loader, and cleared after weight is applied to backend
        self.load_state_dict: bool = False

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
        self.l2_cache_size = l2_cache_size
        logging.info(f"tbe_unique_id: {tbe_unique_id}")
        if self.backend_type == BackendType.SSD:
            logging.info(
                f"Logging SSD offloading setup, tbe_unique_id:{tbe_unique_id}, l2_cache_size:{l2_cache_size}GB, "
                f"enable_async_update:{enable_async_update}, passed_in_path={ssd_directory}, "
                f"num_shards={ssd_rocksdb_shards}, num_threads={ssd_rocksdb_shards}, "
                f"memtable_flush_period={ssd_memtable_flush_period}, memtable_flush_offset={ssd_memtable_flush_offset}, "
                f"l0_files_per_compact={ssd_l0_files_per_compact}, max_D={self.max_D}, "
                f"cache_row_size={self.cache_row_dim}, rate_limit_mbps={ssd_rate_limit_mbps}, "
                f"size_ratio={ssd_size_ratio}, compaction_trigger={ssd_compaction_trigger}, "
                f"lazy_bulk_init_enabled={lazy_bulk_init_enabled}, write_buffer_size_per_tbe={ssd_rocksdb_write_buffer_size}, "
                f"max_write_buffer_num_per_db_shard={ssd_max_write_buffer_num}, "
                f"uniform_init_lower={ssd_uniform_init_lower}, uniform_init_upper={ssd_uniform_init_upper}, "
                f"row_storage_bitwidth={weights_precision.bit_rate()}, block_cache_size_per_tbe={ssd_block_cache_size_per_tbe}, "
                f"use_passed_in_path:{use_passed_in_path}, real_path will be printed in EmbeddingRocksDB, "
                f"enable_raw_embedding_streaming:{self.enable_raw_embedding_streaming}, flushing_block_size:{flushing_block_size}"
            )
            # pyre-fixme[4]: Attribute must be annotated.
            self._ssd_db = torch.classes.fbgemm.EmbeddingRocksDBWrapper(
                ssd_directory,
                ssd_rocksdb_shards,
                ssd_rocksdb_shards,
                ssd_memtable_flush_period,
                ssd_memtable_flush_offset,
                ssd_l0_files_per_compact,
                self.cache_row_dim,
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
                self.enable_raw_embedding_streaming,
                self.res_params.res_store_shards,
                self.res_params.res_server_port,
                self.res_params.table_names,
                self.res_params.table_offsets,
                self.res_params.table_sizes,
                (
                    tensor_pad4(self.table_dims)
                    if self.enable_optimizer_offloading
                    else None
                ),
                (
                    self.table_hash_size_cumsum.cpu()
                    if self.enable_optimizer_offloading
                    else None
                ),
                flushing_block_size,
            )
            if self.bulk_init_chunk_size > 0:
                self.ssd_uniform_init_lower: float = ssd_uniform_init_lower
                self.ssd_uniform_init_upper: float = ssd_uniform_init_upper
                if lazy_bulk_init_enabled:
                    self._lazy_initialize_ssd_tbe()
                else:
                    self._insert_all_kv()
        elif self.backend_type == BackendType.PS:
            self._ssd_db = torch.classes.fbgemm.EmbeddingParameterServerWrapper(
                [host[0] for host in ps_hosts],  # pyre-ignore
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
                self.cache_row_dim,
            )
        elif self.backend_type == BackendType.DRAM:
            logging.info(
                f"Logging DRAM offloading setup, tbe_unique_id:{tbe_unique_id}, l2_cache_size:{l2_cache_size}GB,"
                f"num_shards={ssd_rocksdb_shards},num_threads={ssd_rocksdb_shards},"
                f"max_D={self.max_D},"
                f"uniform_init_lower={ssd_uniform_init_lower},uniform_init_upper={ssd_uniform_init_upper},"
                f"row_storage_bitwidth={weights_precision.bit_rate()},"
                f"self.cache_row_dim={self.cache_row_dim},"
                f"enable_optimizer_offloading={self.enable_optimizer_offloading},"
                f"feature_dims={self.feature_dims},"
                f"hash_size_cumsum={self.hash_size_cumsum},"
                f"backend_return_whole_row={self.backend_return_whole_row}"
            )
            table_dims = (
                tensor_pad4(self.table_dims)
                if self.enable_optimizer_offloading
                else None
            )  # table_dims
            eviction_config = None
            if self.kv_zch_params and self.kv_zch_params.eviction_policy:
                eviction_mem_threshold_gb = (
                    self.kv_zch_params.eviction_policy.eviction_mem_threshold_gb
                    if self.kv_zch_params.eviction_policy.eviction_mem_threshold_gb
                    else self.l2_cache_size
                )
                eviction_config = torch.classes.fbgemm.FeatureEvictConfig(
                    self.kv_zch_params.eviction_policy.eviction_trigger_mode,  # eviction is disabled, 0: disabled, 1: iteration, 2: mem_util, 3: manual
                    self.kv_zch_params.eviction_policy.eviction_strategy,  # evict_trigger_strategy: 0: timestamp, 1: counter (feature score), 2: counter (feature score) + timestamp, 3: feature l2 norm
                    self.kv_zch_params.eviction_policy.eviction_step_intervals,  # trigger_step_interval if trigger mode is iteration
                    eviction_mem_threshold_gb,  # mem_util_threshold_in_GB if trigger mode is mem_util
                    self.kv_zch_params.eviction_policy.ttls_in_mins,  # ttls_in_mins for each table if eviction strategy is timestamp
                    self.kv_zch_params.eviction_policy.counter_thresholds,  # counter_thresholds for each table if eviction strategy is feature score
                    self.kv_zch_params.eviction_policy.counter_decay_rates,  # counter_decay_rates for each table if eviction strategy is feature score
                    self.kv_zch_params.eviction_policy.l2_weight_thresholds,  # l2_weight_thresholds for each table if eviction strategy is feature l2 norm
                    table_dims.tolist() if table_dims is not None else None,
                    self.kv_zch_params.eviction_policy.interval_for_insufficient_eviction_s,
                    self.kv_zch_params.eviction_policy.interval_for_sufficient_eviction_s,
                )
            self._ssd_db = torch.classes.fbgemm.DramKVEmbeddingCacheWrapper(
                self.cache_row_dim,
                ssd_uniform_init_lower,
                ssd_uniform_init_upper,
                eviction_config,
                ssd_rocksdb_shards,  # num_shards
                ssd_rocksdb_shards,  # num_threads
                weights_precision.bit_rate(),  # row_storage_bitwidth
                table_dims,
                (
                    self.table_hash_size_cumsum.cpu()
                    if self.enable_optimizer_offloading
                    else None
                ),  # hash_size_cumsum
                self.backend_return_whole_row,  # backend_return_whole_row
            )
        else:
            raise AssertionError(f"Invalid backend type {self.backend_type}")

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

        if self.enable_raw_embedding_streaming:
            # RES reuse the eviction stream
            self.ssd_event_cache_streamed: torch.cuda.streams.Event = torch.cuda.Event()
            self.ssd_event_cache_streaming_synced: torch.cuda.streams.Event = (
                torch.cuda.Event()
            )
            self.ssd_event_cache_streaming_computed: torch.cuda.streams.Event = (
                torch.cuda.Event()
            )
            self.ssd_event_sp_streamed: torch.cuda.streams.Event = torch.cuda.Event()

            # Updated buffers
            self.register_buffer(
                "lxu_cache_updated_weights",
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

            # For storing embedding indices to update to
            self.register_buffer(
                "lxu_cache_updated_indices",
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

            # For storing the number of updated rows
            self.register_buffer(
                "lxu_cache_updated_count",
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

            # (Indices, Count)
            self.prefetched_info: List[Tuple[Tensor, Tensor]] = []

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

        self.learning_rate_tensor: torch.Tensor = torch.tensor(
            learning_rate, device=torch.device("cpu"), dtype=torch.float32
        )

        self.optimizer_args = invokers.lookup_args_ssd.OptimizerArgs(
            stochastic_rounding=stochastic_rounding,
            gradient_clipping=gradient_clipping,
            max_gradient=max_gradient,
            max_norm=max_norm,
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

        # Create the optimizer state tensors
        for template in self.optimizer.ssd_state_splits(
            self.embedding_specs,
            self.optimizer_state_dtypes,
            self.enable_optimizer_offloading,
        ):
            # pyre-fixme[6]: For 3rd argument expected `Type[dtype]` but got `dtype`.
            self._apply_split(*template)

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
        self.dram_kv_actual_used_chunk_bytes_stats_name: str = (
            f"dram_kv.mem.tbe_id{tbe_unique_id}.actual_used_chunk_bytes"
        )
        self.dram_kv_allocated_bytes_stats_name: str = (
            f"dram_kv.mem.tbe_id{tbe_unique_id}.allocated_bytes"
        )
        self.eviction_sum_evicted_counts_stats_name: str = (
            f"eviction.tbe_id.{tbe_unique_id}.sum_evicted_counts"
        )
        self.eviction_sum_processed_counts_stats_name: str = (
            f"eviction.tbe_id.{tbe_unique_id}.sum_processed_counts"
        )
        self.eviction_evict_rate_stats_name: str = (
            f"eviction.tbe_id.{tbe_unique_id}.evict_rate"
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
            self.stats_reporter.register_stats(self.l2_num_cache_lookups_stats_name)
            self.stats_reporter.register_stats(self.l2_num_cache_evictions_stats_name)
            self.stats_reporter.register_stats(self.l2_cache_free_mem_stats_name)
            self.stats_reporter.register_stats(self.l2_cache_capacity_stats_name)
            self.stats_reporter.register_stats(self.dram_kv_allocated_bytes_stats_name)
            self.stats_reporter.register_stats(
                self.dram_kv_actual_used_chunk_bytes_stats_name
            )
            self.stats_reporter.register_stats(
                self.eviction_sum_evicted_counts_stats_name
            )
            self.stats_reporter.register_stats(
                self.eviction_sum_processed_counts_stats_name
            )
            self.stats_reporter.register_stats(self.eviction_evict_rate_stats_name)
            for t in self.feature_table_map:
                self.stats_reporter.register_stats(
                    f"eviction.feature_table.{t}.evicted_counts"
                )
                self.stats_reporter.register_stats(
                    f"eviction.feature_table.{t}.processed_counts"
                )
                self.stats_reporter.register_stats(
                    f"eviction.feature_table.{t}.evict_rate"
                )
            self.stats_reporter.register_stats(
                "eviction.feature_table.full_duration_ms"
            )
            self.stats_reporter.register_stats(
                "eviction.feature_table.exec_duration_ms"
            )
            self.stats_reporter.register_stats(
                "eviction.feature_table.exec_div_full_duration_rate"
            )

        self.bounds_check_version: int = get_bounds_check_version_for_platform()

    @cached_property
    def cache_row_dim(self) -> int:
        """
        Compute the effective physical cache row size taking into account
        padding to the nearest 4 elements and the optimizer state appended to
        the back of the row
        """
        if self.enable_optimizer_offloading:
            return self.max_D + pad4(
                # Compute the number of elements of cache_dtype needed to store
                # the optimizer state
                self.optimizer_state_dim
            )
        else:
            return self.max_D

    @cached_property
    def optimizer_state_dim(self) -> int:
        return int(
            math.ceil(
                self.optimizer.state_size_nbytes(
                    self.max_D, self.optimizer_state_dtypes
                )
                / self.weights_precision.as_dtype().itemsize
            )
        )

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
        self._ssd_db.toggle_compaction(False)
        row_offset = 0
        row_count = floor(
            self.bulk_init_chunk_size
            / (self.cache_row_dim * self.weights_precision.as_dtype().itemsize)
        )
        total_dim0 = 0
        for dim0, _ in self.embedding_specs:
            total_dim0 += dim0

        start_ts = time.time()
        # TODO: do we have case for non-kvzch ssd with bulk init enabled + optimizer offloading? probably not?
        #       if we have such cases, we should only init the emb dim not the optimizer dim
        chunk_tensor = torch.empty(
            row_count,
            self.cache_row_dim,
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
        end_ts = time.time()
        elapsed = int((end_ts - start_ts) * 1e6)
        logging.info(
            f"TBE bulk initialization took {elapsed:_} us, bulk_init_chunk_size={self.bulk_init_chunk_size}, each batch of {row_count} rows, total rows of {total_dim0}"
        )
        self._ssd_db.toggle_compaction(True)

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
        if time_unit == "us":
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
            is_bwd (bool): A flag to indicate if the eviction is during backward
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
                    indices_cpu.cpu(),
                    rows_cpu,
                    actions_count_cpu,
                    self.timestep,
                    is_bwd,
                )

                if post_event is not None:
                    stream.record_event(post_event)

    def raw_embedding_stream_sync(
        self,
        stream: torch.cuda.Stream,
        pre_event: Optional[torch.cuda.Event],
        post_event: Optional[torch.cuda.Event],
        name: Optional[str] = "",
    ) -> None:
        """
        Blocking wait the copy operation of the tensors to be streamed,
        to make sure they are not overwritten
        Args:
            stream (Stream): The CUDA stream that cudaStreamAddCallback will
                             synchronize the host function with.  Moreover, the
                             asynchronous D->H memory copies will operate on
                             this stream
            pre_event (Event): The CUDA event that the stream has to wait on
            post_event (Event): The CUDA event that the current will record on
                                when the eviction is done
        Returns:
            None
        """
        with record_function(f"## ssd_stream_{name} ##"):
            with torch.cuda.stream(stream):
                if pre_event is not None:
                    stream.wait_event(pre_event)

                self.record_function_via_dummy_profile(
                    f"## ssd_stream_sync_{name} ##",
                    self.ssd_db.stream_sync_cuda,
                )

                if post_event is not None:
                    stream.record_event(post_event)

    def raw_embedding_stream(
        self,
        rows: Tensor,
        indices_cpu: Tensor,
        actions_count_cpu: Tensor,
        stream: torch.cuda.Stream,
        pre_event: Optional[torch.cuda.Event],
        post_event: Optional[torch.cuda.Event],
        is_rows_uvm: bool,
        blocking_tensor_copy: bool = True,
        name: Optional[str] = "",
    ) -> None:
        """
        Stream data from the given input tensors to a remote service
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
        with record_function(f"## ssd_stream_{name} ##"):
            with torch.cuda.stream(stream):
                if pre_event is not None:
                    stream.wait_event(pre_event)

                rows_cpu = rows if is_rows_uvm else self.to_pinned_cpu(rows)

                rows.record_stream(stream)

                self.record_function_via_dummy_profile(
                    f"## ssd_stream_{name} ##",
                    self.ssd_db.stream_cuda,
                    indices_cpu,
                    rows_cpu,
                    actions_count_cpu,
                    blocking_tensor_copy,
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

            if self.enable_raw_embedding_streaming:
                self.raw_embedding_stream(
                    rows=inserted_rows,
                    indices_cpu=post_bwd_evicted_indices_cpu,
                    actions_count_cpu=actions_count_cpu,
                    stream=self.ssd_eviction_stream,
                    pre_event=self.ssd_event_backward,
                    post_event=self.ssd_event_sp_streamed,
                    is_rows_uvm=True,
                    blocking_tensor_copy=True,
                    name="scratch_pad",
                )
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
            has_raw_embedding_streaming = False
            if self.enable_raw_embedding_streaming:
                # when pipelining is enabled
                # prefetch in iter i happens before the backward sparse in iter i - 1
                # so embeddings for iter i - 1's changed ids are not updated.
                # so we can only fetch the indices from the iter i - 2
                # when pipelining is disabled
                # prefetch in iter i happens before forward iter i
                # so we can get the iter i - 1's changed ids safely.
                target_prev_iter = 1
                if self.prefetch_pipeline:
                    target_prev_iter = 2
                if len(self.prefetched_info) > (target_prev_iter - 1):
                    with record_function(
                        "## ssd_lookup_prefetched_rows {} {} ##".format(
                            self.timestep, self.tbe_unique_id
                        )
                    ):
                        # wait for the copy to finish before overwriting the buffer
                        self.raw_embedding_stream_sync(
                            stream=self.ssd_eviction_stream,
                            pre_event=self.ssd_event_cache_streamed,
                            post_event=self.ssd_event_cache_streaming_synced,
                            name="cache_update",
                        )
                        current_stream.wait_event(self.ssd_event_cache_streaming_synced)
                        (updated_indices, updated_counts_gpu) = (
                            self.prefetched_info.pop(0)
                        )
                        self.lxu_cache_updated_indices[: updated_indices.size(0)].copy_(
                            updated_indices,
                            non_blocking=True,
                        )
                        self.lxu_cache_updated_count[:1].copy_(
                            updated_counts_gpu, non_blocking=True
                        )
                        has_raw_embedding_streaming = True

                with record_function(
                    "## ssd_save_prefetched_rows {} {} ##".format(
                        self.timestep, self.tbe_unique_id
                    )
                ):
                    masked_updated_indices = torch.where(
                        torch.where(lxu_cache_locations != -1, True, False),
                        linear_cache_indices,
                        -1,
                    )

                    (
                        uni_updated_indices,
                        uni_updated_indices_length,
                    ) = get_unique_indices_v2(
                        masked_updated_indices,
                        self.total_hash_size,
                        compute_count=False,
                        compute_inverse_indices=False,
                    )
                    assert uni_updated_indices is not None
                    assert uni_updated_indices_length is not None
                    # The unique indices has 1 more -1 element than needed,
                    # which might make the tensor length go out of range
                    # compared to the pre-allocated buffer.
                    unique_len = min(
                        self.lxu_cache_weights.size(0),
                        uni_updated_indices.size(0),
                    )
                    self.prefetched_info.append(
                        (
                            uni_updated_indices.narrow(0, 0, unique_len),
                            uni_updated_indices_length.clamp(max=unique_len),
                        )
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
            inserted_rows_shape = (assigned_cache_slots.numel(), self.cache_row_dim)
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

            if self.enable_raw_embedding_streaming and has_raw_embedding_streaming:
                current_stream.wait_event(self.ssd_event_sp_streamed)
                with record_function(
                    "## ssd_compute_updated_rows {} {} ##".format(
                        self.timestep, self.tbe_unique_id
                    )
                ):
                    # cache rows that are changed in the previous iteration
                    updated_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
                        self.lxu_cache_updated_indices,
                        self.lxu_cache_state,
                        self.total_hash_size,
                        self.gather_ssd_cache_stats,
                        self.local_ssd_cache_stats,
                    )
                    torch.ops.fbgemm.masked_index_select(
                        self.lxu_cache_updated_weights,
                        updated_cache_locations,
                        self.lxu_cache_weights,
                        self.lxu_cache_updated_count,
                    )
                current_stream.record_event(self.ssd_event_cache_streaming_computed)

                self.raw_embedding_stream(
                    rows=self.lxu_cache_updated_weights,
                    indices_cpu=self.lxu_cache_updated_indices,
                    actions_count_cpu=self.lxu_cache_updated_count,
                    stream=self.ssd_eviction_stream,
                    pre_event=self.ssd_event_cache_streaming_computed,
                    post_event=self.ssd_event_cache_streamed,
                    is_rows_uvm=True,
                    blocking_tensor_copy=False,
                    name="cache_update",
                )

            if self.gather_ssd_cache_stats:
                # call to collect past SSD IO dur right before next rocksdb IO

                self.ssd_cache_stats = torch.add(
                    self.ssd_cache_stats, self.local_ssd_cache_stats
                )
                self._report_kv_backend_stats()

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

    def _increment_iteration(self) -> int:
        # Although self.iter_cpu is created on CPU. It might be transferred to
        # GPU by the user. So, we need to transfer it to CPU explicitly. This
        # should be done only once.
        self.iter_cpu = self.iter_cpu.cpu()

        # Sync with loaded state
        # Wrap to make it compatible with PT2 compile
        if not is_torchdynamo_compiling():
            if self.iter_cpu.item() == 0:
                self.iter_cpu.fill_(self.iter.cpu().item())

        # Increment the iteration counter
        # The CPU counterpart is used for local computation
        iter_int = int(self.iter_cpu.add_(1).item())
        # The GPU counterpart is used for checkpointing
        self.iter.add_(1)

        return iter_int

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        # pyre-fixme[7]: Expected `Tensor` but got implicit return value of `None`.
    ) -> Tensor:
        self.clear_cache()
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
            enable_optimizer_offloading=self.enable_optimizer_offloading,
            # pyre-fixme[6]: Expected `lookup_args_ssd.VBEMetadata` but got `lookup_args.VBEMetadata`
            vbe_metadata=vbe_metadata,
            learning_rate_tensor=self.learning_rate_tensor,
            info_B_num_bits=self.info_B_num_bits,
            info_B_mask=self.info_B_mask,
        )

        self.timesteps_prefetched.pop(0)
        self.step += 1

        # Increment the iteration (value is used for certain optimizers)
        iter_int = self._increment_iteration()

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
            return invokers.lookup_rowwise_adagrad_ssd.invoke(
                common_args, self.optimizer_args, momentum1
            )

        elif self.optimizer == OptimType.PARTIAL_ROWWISE_ADAM:
            momentum2 = invokers.lookup_args_ssd.Momentum(
                # pyre-ignore[6]
                dev=self.momentum2_dev,
                # pyre-ignore[6]
                host=self.momentum2_host,
                # pyre-ignore[6]
                uvm=self.momentum2_uvm,
                # pyre-ignore[6]
                offsets=self.momentum2_offsets,
                # pyre-ignore[6]
                placements=self.momentum2_placements,
            )

            return invokers.lookup_partial_rowwise_adam_ssd.invoke(
                common_args, self.optimizer_args, momentum1, momentum2, iter_int
            )

    @torch.jit.ignore
    def _split_optimizer_states_non_kv_zch(
        self,
    ) -> List[List[torch.Tensor]]:
        """
        Returns a list of optimizer states (view), split by table.

        Returns:
            A list of list of states. Shape = (the number of tables, the number
            of states).

            The following shows the list of states (in the returned order) for
            each optimizer:

            (1) `EXACT_ROWWISE_ADAGRAD`: `momentum1` (rowwise)

            (1) `PARTIAL_ROWWISE_ADAM`: `momentum1`, `momentum2` (rowwise)
        """

        # Row count per table
        (rows, dims) = zip(*self.embedding_specs)
        # Cumulative row counts per table for rowwise states
        row_count_cumsum: List[int] = [0] + list(itertools.accumulate(rows))
        # Cumulative element counts per table for elementwise states
        elem_count_cumsum: List[int] = [0] + list(
            itertools.accumulate([r * d for r, d in self.embedding_specs])
        )

        # pyre-ignore[53]
        def _slice(tensor: Tensor, t: int, rowwise: bool) -> Tensor:
            d: int = dims[t]
            e: int = rows[t]

            if not rowwise:
                # Optimizer state is element-wise - compute the table offset for
                # the table, view the slice as 2D tensor
                return tensor.detach()[
                    elem_count_cumsum[t] : elem_count_cumsum[t + 1]
                ].view(-1, d)
            else:
                # Optimizer state is row-wise - fetch elements in range and view
                # slice as 1D
                return tensor.detach()[
                    row_count_cumsum[t] : row_count_cumsum[t + 1]
                ].view(e)

        if self.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
            return [
                [_slice(self.momentum1_dev, t, rowwise=True)]
                for t, _ in enumerate(rows)
            ]
        elif self.optimizer == OptimType.PARTIAL_ROWWISE_ADAM:
            return [
                [
                    _slice(self.momentum1_dev, t, rowwise=False),
                    # pyre-ignore[6]
                    _slice(self.momentum2_dev, t, rowwise=True),
                ]
                for t, _ in enumerate(rows)
            ]
        else:
            raise NotImplementedError(
                f"Getting optimizer states is not supported for {self.optimizer}"
            )

    @torch.jit.ignore
    def _split_optimizer_states_kv_zch_no_offloading(
        self,
        sorted_ids: torch.Tensor,
    ) -> List[List[torch.Tensor]]:

        # Row count per table
        (rows, dims) = zip(*self.embedding_specs)
        # Cumulative row counts per table for rowwise states
        row_count_cumsum: List[int] = [0] + list(itertools.accumulate(rows))
        # Cumulative element counts per table for elementwise states
        elem_count_cumsum: List[int] = [0] + list(
            itertools.accumulate([r * d for r, d in self.embedding_specs])
        )

        # pyre-ignore[53]
        def _slice(state_name: str, tensor: Tensor, t: int, rowwise: bool) -> Tensor:
            d: int = dims[t]

            # pyre-ignore[16]
            bucket_id_start, _ = self.kv_zch_params.bucket_offsets[t]
            # pyre-ignore[16]
            bucket_size = self.kv_zch_params.bucket_sizes[t]

            if sorted_ids is None or sorted_ids[t].numel() == 0:
                # Empty optimizer state for module initialization
                return torch.empty(
                    0,
                    dtype=(
                        self.optimizer_state_dtypes.get(
                            state_name, SparseType.FP32
                        ).as_dtype()
                    ),
                    device="cpu",
                )

            elif not rowwise:
                # Optimizer state is element-wise - materialize the local ids
                # based on the sorted_ids compute the table offset for the
                # table, view the slice as 2D tensor of e x d, then fetch the
                # sub-slice by local ids
                #
                # local_ids is [N, 1], flatten it to N to keep the returned tensor 2D
                local_ids = (sorted_ids[t] - bucket_id_start * bucket_size).view(-1)
                return (
                    tensor.detach()
                    .cpu()[elem_count_cumsum[t] : elem_count_cumsum[t + 1]]
                    .view(-1, d)[local_ids]
                )

            else:
                # Optimizer state is row-wise - materialize the local ids based
                # on the sorted_ids and table offset (i.e. row count cumsum),
                # then fetch by local ids
                linearized_local_ids = (
                    sorted_ids[t] - bucket_id_start * bucket_size + row_count_cumsum[t]
                )
                return tensor.detach().cpu()[linearized_local_ids].view(-1)

        if self.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
            return [
                [_slice("momentum1", self.momentum1_dev, t, rowwise=True)]
                for t, _ in enumerate(rows)
            ]

        elif self.optimizer == OptimType.PARTIAL_ROWWISE_ADAM:
            return [
                [
                    _slice("momentum1", self.momentum1_dev, t, rowwise=False),
                    # pyre-ignore[6]
                    _slice("momentum2", self.momentum2_dev, t, rowwise=True),
                ]
                for t, _ in enumerate(rows)
            ]

        else:
            raise NotImplementedError(
                f"Getting optimizer states is not supported for {self.optimizer}"
            )

    @torch.jit.ignore
    def _split_optimizer_states_kv_zch_w_offloading(
        self,
        sorted_ids: torch.Tensor,
        no_snapshot: bool = True,
        should_flush: bool = False,
    ) -> List[List[torch.Tensor]]:
        dtype = self.weights_precision.as_dtype()
        # Row count per table
        (rows_, dims_) = zip(*self.embedding_specs)
        # Cumulative row counts per table for rowwise states
        row_count_cumsum: List[int] = [0] + list(itertools.accumulate(rows_))

        snapshot_handle, _ = self._may_create_snapshot_for_state_dict(
            no_snapshot=no_snapshot,
            should_flush=should_flush,
        )

        # pyre-ignore[53]
        def _fetch_offloaded_optimizer_states(
            t: int,
        ) -> List[Tensor]:
            e: int = rows_[t]
            d: int = dims_[t]

            # pyre-ignore[16]
            bucket_id_start, _ = self.kv_zch_params.bucket_offsets[t]
            # pyre-ignore[16]
            bucket_size = self.kv_zch_params.bucket_sizes[t]

            row_offset = row_count_cumsum[t] - (bucket_id_start * bucket_size)
            # Count of rows to fetch
            rows_to_fetch = sorted_ids[t].numel()

            # Lookup the byte offsets for each optimizer state
            optimizer_state_byte_offsets = self.optimizer.byte_offsets_along_row(
                d, self.weights_precision, self.optimizer_state_dtypes
            )
            # Find the minimum start of all the start/end pairs - we have to
            # offset the start/end pairs by this value to get the correct start/end
            offset_ = min(
                [start for _, (start, _) in optimizer_state_byte_offsets.items()]
            )
            # Update the start/end pairs to be relative to offset_
            optimizer_state_byte_offsets = dict(
                (k, (v1 - offset_, v2 - offset_))
                for k, (v1, v2) in optimizer_state_byte_offsets.items()
            )

            # Since the backend returns cache rows that pack the weights and
            # optimizer states together, reading the whole tensor could cause OOM,
            # so we use the KVTensorWrapper abstraction to query the backend and
            # fetch the data in chunks instead.
            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                shape=[
                    e,
                    # Dim is terms of **weights** dtype
                    self.optimizer_state_dim,
                ],
                dtype=dtype,
                row_offset=row_offset,
                snapshot_handle=snapshot_handle,
                sorted_indices=sorted_ids[t],
                width_offset=pad4(d),
            )
            (
                tensor_wrapper.set_embedding_rocks_dp_wrapper(self.ssd_db)
                if self.backend_type == BackendType.SSD
                else tensor_wrapper.set_dram_db_wrapper(self.ssd_db)
            )

            # Fetch the state size table for the given weights domension
            state_size_table = self.optimizer.state_size_table(d)

            # Create a 2D output buffer of [rows x optimizer state dim] with the
            # weights type as the type.  For optimizers with multiple states (e.g.
            # momentum1 and momentum2), this tensor will include data from all
            # states, hence self.optimizer_state_dim as the row size.
            optimizer_states_buffer = torch.empty(
                (rows_to_fetch, self.optimizer_state_dim), dtype=dtype, device="cpu"
            )

            # Set the chunk size for fetching
            chunk_size = (
                # 10M rows => 260(max_D)* 2(ele_bytes) * 10M => 5.2GB mem spike
                10_000_000
            )
            logging.info(f"split optimizer chunk rows: {chunk_size}")

            # Chunk the fetching by chunk_size
            for i in range(0, rows_to_fetch, chunk_size):
                length = min(chunk_size, rows_to_fetch - i)

                # Fetch from backend and copy to the output buffer
                optimizer_states_buffer.narrow(0, i, length).copy_(
                    tensor_wrapper.narrow(0, i, length).view(dtype)
                )

            # Now split up the buffer into N views, N for each optimizer state
            optimizer_states: List[Tensor] = []
            for state_name in self.optimizer.state_names():
                # Extract the offsets
                (start, end) = optimizer_state_byte_offsets[state_name]

                state = optimizer_states_buffer.view(
                    # Force tensor to byte view
                    dtype=torch.uint8
                    # Copy by byte offsets
                )[:, start:end].view(
                    # Re-view in the state's target dtype
                    self.optimizer_state_dtypes.get(
                        state_name, SparseType.FP32
                    ).as_dtype()
                )

                optimizer_states.append(
                    # If the state is rowwise (i.e. just one element per row),
                    # then re-view as 1D tensor
                    state
                    if state_size_table[state_name] > 1
                    else state.view(-1)
                )

            # Return the views
            return optimizer_states

        return [
            (
                self.optimizer.empty_states([0], [d], self.optimizer_state_dtypes)[0]
                # Return a set of empty states if sorted_ids[t] is empty
                if sorted_ids is None or sorted_ids[t].numel() == 0
                # Else fetch the list of optimizer states for the table
                else _fetch_offloaded_optimizer_states(t)
            )
            for t, d in enumerate(dims_)
        ]

    @torch.jit.ignore
    def _split_optimizer_states_kv_zch_whole_row(
        self,
        sorted_ids: torch.Tensor,
        no_snapshot: bool = True,
        should_flush: bool = False,
    ) -> List[List[torch.Tensor]]:
        dtype = self.weights_precision.as_dtype()

        # Row and dimension counts per table
        # rows_ is only used here to compute the virtual table offsets
        (rows_, dims_) = zip(*self.embedding_specs)

        # Cumulative row counts per (virtual) table for rowwise states
        row_count_cumsum: List[int] = [0] + list(itertools.accumulate(rows_))

        snapshot_handle, _ = self._may_create_snapshot_for_state_dict(
            no_snapshot=no_snapshot,
            should_flush=should_flush,
        )

        # pyre-ignore[53]
        def _fetch_offloaded_optimizer_states(
            t: int,
        ) -> List[Tensor]:
            d: int = dims_[t]

            # pyre-ignore[16]
            bucket_id_start, _ = self.kv_zch_params.bucket_offsets[t]
            # pyre-ignore[16]
            bucket_size = self.kv_zch_params.bucket_sizes[t]
            row_offset = row_count_cumsum[t] - (bucket_id_start * bucket_size)

            # When backend returns whole row, the optimizer will be returned as
            # PMT directly
            if sorted_ids[t].size(0) == 0 and self.local_weight_counts[t] > 0:
                logging.info(
                    f"Before opt PMT loading, resetting id tensor with {self.local_weight_counts[t]}"
                )
                sorted_ids[t] = torch.zeros(
                    (self.local_weight_counts[t], 1),
                    device=torch.device("cpu"),
                    dtype=torch.int64,
                )

            # Lookup the byte offsets for each optimizer state relative to the
            # start of the weights
            optimizer_state_byte_offsets = self.optimizer.byte_offsets_along_row(
                d, self.weights_precision, self.optimizer_state_dtypes
            )
            # Get the number of elements (of the optimizer state dtype) per state
            optimizer_state_size_table = self.optimizer.state_size_table(d)

            # Get metaheader dimensions in number of elements of weight dtype
            metaheader_dim = (
                # pyre-ignore[16]
                self.kv_zch_params.eviction_policy.meta_header_lens[t]
            )

            # Now split up the buffer into N views, N for each optimizer state
            optimizer_states: List[PartiallyMaterializedTensor] = []
            for state_name in self.optimizer.state_names():
                state_dtype = self.optimizer_state_dtypes.get(
                    state_name, SparseType.FP32
                ).as_dtype()

                # Get the size of the state in elements of the optimizer state,
                # in terms of the **weights** dtype
                state_size = math.ceil(
                    optimizer_state_size_table[state_name]
                    * state_dtype.itemsize
                    / dtype.itemsize
                )

                # Extract the offsets relative to the start of the weights (in
                # num bytes)
                (start, _) = optimizer_state_byte_offsets[state_name]

                # Convert the start to number of elements in terms of the
                # **weights** dtype, then add the mmetaheader dim offset
                start = metaheader_dim + start // dtype.itemsize

                shape = [
                    (
                        sorted_ids[t].size(0)
                        if sorted_ids is not None and sorted_ids[t].size(0) > 0
                        else self.local_weight_counts[t]
                    ),
                    (
                        # Dim is in terms of the **weights** dtype
                        state_size
                    ),
                ]

                # NOTE: We have to view using the **weights** dtype, as
                # there is currently a bug with KVTensorWrapper where using
                # a different dtype does not result in the same bytes being
                # returned, e.g.
                #
                # KVTensorWrapper(dtype=fp32, width_offset=130, shape=[N, 1])
                #
                # is NOT the same as
                #
                # KVTensorWrapper(dtype=fp16, width_offset=260, shape=[N, 2]).view(-1).view(fp32)
                #
                # TODO: Fix KVTensorWrapper to support viewing data under different dtypes
                tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                    shape=shape,
                    dtype=(
                        # NOTE: Use the *weights* dtype
                        dtype
                    ),
                    row_offset=row_offset,
                    snapshot_handle=snapshot_handle,
                    sorted_indices=sorted_ids[t],
                    width_offset=(
                        # NOTE: Width offset is in terms of **weights** dtype
                        start
                    ),
                    # Optimizer written to DB with weights, so skip write here
                    read_only=True,
                )
                (
                    tensor_wrapper.set_embedding_rocks_dp_wrapper(self.ssd_db)
                    if self.backend_type == BackendType.SSD
                    else tensor_wrapper.set_dram_db_wrapper(self.ssd_db)
                )

                optimizer_states.append(
                    PartiallyMaterializedTensor(tensor_wrapper, True)
                )

            # pyre-ignore [7]
            return optimizer_states

        return [_fetch_offloaded_optimizer_states(t) for t, _ in enumerate(dims_)]

    @torch.jit.export
    def split_optimizer_states(
        self,
        sorted_id_tensor: Optional[List[torch.Tensor]] = None,
        no_snapshot: bool = True,
        should_flush: bool = False,
    ) -> List[List[torch.Tensor]]:
        """
        Returns a list of optimizer states split by table.

        Since EXACT_ROWWISE_ADAGRAD has small optimizer states, we would generate
        a full tensor for each table (shard). When other optimizer types are supported,
        we should integrate with KVTensorWrapper (ssd_split_table_batched_embeddings.cpp)
        to allow caller to read the optimizer states using `narrow()` in a rolling-window manner.

        Args:
            sorted_id_tensor (Optional[List[torch.Tensor]]): sorted id tensor by table, used to query optimizer
            state from backend. Call should reuse the generated id tensor from weight state_dict, to guarantee
            id consistency between weight and optimizer states.

        """

        # Handle the non-KVZCH case
        if not self.kv_zch_params:
            # If not in KV
            return self._split_optimizer_states_non_kv_zch()

        # Handle the loading-from-state-dict case
        if self.load_state_dict:
            # Initialize for checkpointing loading
            assert (
                self._cached_kvzch_data is not None
                and self._cached_kvzch_data.cached_optimizer_states_per_table
            ), "optimizer state is not initialized for load checkpointing"

            return self._cached_kvzch_data.cached_optimizer_states_per_table

        logging.info(
            f"split_optimizer_states for KV ZCH: {no_snapshot=}, {should_flush=}"
        )
        start_time = time.time()

        if not self.enable_optimizer_offloading:
            # Handle the KVZCH non-optimizer-offloading case
            optimizer_states = self._split_optimizer_states_kv_zch_no_offloading(
                sorted_id_tensor
            )

        elif not self.backend_return_whole_row:
            # Handle the KVZCH with-optimizer-offloading case
            optimizer_states = self._split_optimizer_states_kv_zch_w_offloading(
                sorted_id_tensor, no_snapshot, should_flush
            )

        else:
            # Handle the KVZCH with-optimizer-offloading backend-whole-row case
            optimizer_states = self._split_optimizer_states_kv_zch_whole_row(
                sorted_id_tensor, no_snapshot, should_flush
            )

        logging.info(
            f"KV ZCH tables split_optimizer_states query latency: {(time.time() - start_time) * 1000} ms, "
            # pyre-ignore[16]
            f"num ids list: {None if not sorted_id_tensor else [ids.numel() for ids in sorted_id_tensor]}"
        )

        return optimizer_states

    @torch.jit.export
    def get_optimizer_state(
        self,
        sorted_id_tensor: Optional[List[torch.Tensor]],
        no_snapshot: bool = True,
        should_flush: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Returns a list of dictionaries of optimizer states split by table.
        """
        states_list: List[List[Tensor]] = self.split_optimizer_states(
            sorted_id_tensor=sorted_id_tensor,
            no_snapshot=no_snapshot,
            should_flush=should_flush,
        )
        state_names = self.optimizer.state_names()
        return [dict(zip(state_names, states)) for states in states_list]

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

        for t, (row, _) in enumerate(self.embedding_specs):
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

    def clear_cache(self) -> None:
        # clear KV ZCH cache for checkpointing
        self._cached_kvzch_data = None

    @torch.jit.ignore
    # pyre-ignore [3] - do not definte snapshot class EmbeddingSnapshotHandleWrapper to avoid import dependency in other production code
    def _may_create_snapshot_for_state_dict(
        self,
        no_snapshot: bool = True,
        should_flush: bool = False,
    ):
        """
        Create a rocksdb snapshot if needed.
        """
        start_time = time.time()
        # Force device synchronize for now
        torch.cuda.synchronize()
        snapshot_handle = None
        checkpoint_handle = None
        if self.backend_type == BackendType.SSD:
            # Create a rocksdb snapshot
            if not no_snapshot:
                # Flush L1 and L2 caches
                self.flush(force=should_flush)
                logging.info(
                    f"flush latency for weight states: {(time.time() - start_time) * 1000} ms"
                )
                snapshot_handle = self.ssd_db.create_snapshot()
                checkpoint_handle = self.ssd_db.get_active_checkpoint_uuid(self.step)
                logging.info(
                    f"created snapshot for weight states: {snapshot_handle}, latency: {(time.time() - start_time) * 1000} ms"
                )
        elif self.backend_type == BackendType.DRAM:
            # if there is any ongoing eviction, lets wait until eviction is finished before state_dict
            # so that we can reach consistent model state before/after state_dict
            evict_wait_start_time = time.time()
            self.ssd_db.wait_until_eviction_done()
            logging.info(
                f"state_dict wait for ongoing eviction: {time.time() - evict_wait_start_time} s"
            )
            self.flush(force=should_flush)
        return snapshot_handle, checkpoint_handle

    @torch.jit.export
    def split_embedding_weights(
        self,
        no_snapshot: bool = True,
        should_flush: bool = False,
    ) -> Tuple[  # TODO: make this a NamedTuple for readability
        Union[List[PartiallyMaterializedTensor], List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
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
            tuples of 3 lists, each element corresponds to a logical table
            1st arg: partially materialized tensors, each representing a table
            2nd arg: input id sorted in bucket id ascending order
            3rd arg: active id count per bucket id, tensor size is [bucket_id_end - bucket_id_start]
                    where for the i th element, we have i + bucket_id_start = global bucket id
            4th arg: kvzch eviction metadata for each input id sorted in bucket id ascending order
        """
        snapshot_handle, checkpoint_handle = self._may_create_snapshot_for_state_dict(
            no_snapshot=no_snapshot,
            should_flush=should_flush,
        )

        dtype = self.weights_precision.as_dtype()
        if self.load_state_dict and self.kv_zch_params:
            # init for checkpointing loading
            assert (
                self._cached_kvzch_data is not None
            ), "weight id and bucket state are not initialized for load checkpointing"
            return (
                self._cached_kvzch_data.cached_weight_tensor_per_table,
                self._cached_kvzch_data.cached_id_tensor_per_table,
                self._cached_kvzch_data.cached_bucket_splits,
                [],  # metadata tensor is not needed for checkpointing loading
            )
        start_time = time.time()
        pmt_splits = []
        bucket_sorted_id_splits = [] if self.kv_zch_params else None
        active_id_cnt_per_bucket_split = [] if self.kv_zch_params else None
        metadata_splits = [] if self.kv_zch_params else None

        table_offset = 0
        for i, (emb_height, emb_dim) in enumerate(self.embedding_specs):
            bucket_ascending_id_tensor = None
            bucket_t = None
            metadata_tensor = None
            row_offset = table_offset
            metaheader_dim = 0
            if self.kv_zch_params:
                bucket_id_start, bucket_id_end = self.kv_zch_params.bucket_offsets[i]
                # pyre-ignore
                bucket_size = self.kv_zch_params.bucket_sizes[i]
                metaheader_dim = (
                    # pyre-ignore[16]
                    self.kv_zch_params.eviction_policy.meta_header_lens[i]
                )

                # linearize with table offset
                table_input_id_start = table_offset
                table_input_id_end = table_offset + emb_height
                # 1. get all keys from backend for one table
                unordered_id_tensor = self._ssd_db.get_keys_in_range_by_snapshot(
                    table_input_id_start,
                    table_input_id_end,
                    table_offset,
                    snapshot_handle,
                )
                # 2. sorting keys in bucket ascending order
                bucket_ascending_id_tensor, bucket_t = (
                    torch.ops.fbgemm.get_bucket_sorted_indices_and_bucket_tensor(
                        unordered_id_tensor,
                        0,  # id--bucket hashing mode, 0 for chunk-based hashing, 1 for interleave-based hashing
                        0,  # local bucket offset
                        bucket_id_end - bucket_id_start,  # local bucket num
                        bucket_size,
                    )
                )
                metadata_tensor = self._ssd_db.get_kv_zch_eviction_metadata_by_snapshot(
                    bucket_ascending_id_tensor,
                    torch.as_tensor(bucket_ascending_id_tensor.size(0)),
                    snapshot_handle,
                )

                # 3. convert local id back to global id
                bucket_ascending_id_tensor.add_(bucket_id_start * bucket_size)

                if (
                    bucket_ascending_id_tensor.size(0) == 0
                    and self.local_weight_counts[i] > 0
                ):
                    logging.info(
                        f"before weight PMT loading, resetting id tensor with {self.local_weight_counts[i]}"
                    )
                    bucket_ascending_id_tensor = torch.zeros(
                        (self.local_weight_counts[i], 1),
                        device=torch.device("cpu"),
                        dtype=torch.int64,
                    )
                    metadata_tensor = torch.zeros(
                        (self.local_weight_counts[i], 1),
                        device=torch.device("cpu"),
                        dtype=torch.int64,
                    )
                    # self.local_weight_counts[i] = 0  # Reset the count

                # pyre-ignore [16] bucket_sorted_id_splits is not None
                bucket_sorted_id_splits.append(bucket_ascending_id_tensor)
                active_id_cnt_per_bucket_split.append(bucket_t)
                metadata_splits.append(metadata_tensor)

                # for KV ZCH tbe, the sorted_indices is global id for checkpointing and publishing
                # but in backend, local id is used during training, so the KVTensorWrapper need to convert global id to local id
                # first, then linearize the local id with table offset, the formulat is x + table_offset - local_shard_offset
                # to achieve this, the row_offset will be set to (table_offset - local_shard_offset)
                row_offset = table_offset - (bucket_id_start * bucket_size)

            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                shape=[
                    (
                        bucket_ascending_id_tensor.size(0)
                        if bucket_ascending_id_tensor is not None
                        else emb_height
                    ),
                    (
                        (
                            metaheader_dim  # metaheader is already padded
                            + pad4(emb_dim)
                            + pad4(self.optimizer_state_dim)
                        )
                        if self.backend_return_whole_row
                        else emb_dim
                    ),
                ],
                dtype=dtype,
                row_offset=row_offset,
                snapshot_handle=snapshot_handle,
                # set bucket_ascending_id_tensor to kvt wrapper, so narrow will follow the id order to return
                # embedding weights.
                sorted_indices=(
                    bucket_ascending_id_tensor if self.kv_zch_params else None
                ),
                checkpoint_handle=checkpoint_handle,
            )
            (
                tensor_wrapper.set_embedding_rocks_dp_wrapper(self.ssd_db)
                if self.backend_type == BackendType.SSD
                else tensor_wrapper.set_dram_db_wrapper(self.ssd_db)
            )
            table_offset += emb_height
            pmt_splits.append(
                PartiallyMaterializedTensor(
                    tensor_wrapper,
                    True if self.kv_zch_params else False,
                )
            )
        logging.info(
            f"split_embedding_weights latency: {(time.time() - start_time) * 1000} ms, "
        )
        if self.kv_zch_params is not None:
            logging.info(
                # pyre-ignore [16]
                f"num ids list: {[ids.numel() for ids in bucket_sorted_id_splits]}"
            )

        return (
            pmt_splits,
            bucket_sorted_id_splits,
            active_id_cnt_per_bucket_split,
            metadata_splits,
        )

    @torch.jit.ignore
    def _apply_state_dict_w_offloading(self) -> None:
        # Row count per table
        (rows, _) = zip(*self.embedding_specs)
        # Cumulative row counts per table for rowwise states
        row_count_cumsum: List[int] = [0] + list(itertools.accumulate(rows))

        for t, _ in enumerate(self.embedding_specs):
            # pyre-ignore [16]
            bucket_id_start, _ = self.kv_zch_params.bucket_offsets[t]
            # pyre-ignore [16]
            bucket_size = self.kv_zch_params.bucket_sizes[t]
            row_offset = row_count_cumsum[t] - bucket_id_start * bucket_size

            # pyre-ignore [16]
            weight_state = self._cached_kvzch_data.cached_weight_tensor_per_table[t]
            # pyre-ignore [16]
            opt_states = self._cached_kvzch_data.cached_optimizer_states_per_table[t]

            self.streaming_write_weight_and_id_per_table(
                weight_state,
                opt_states,
                # pyre-ignore [16]
                self._cached_kvzch_data.cached_id_tensor_per_table[t],
                row_offset,
            )
            self._cached_kvzch_data.cached_weight_tensor_per_table[t] = None
            self._cached_kvzch_data.cached_optimizer_states_per_table[t] = None

    @torch.jit.ignore
    def _apply_state_dict_no_offloading(self) -> None:
        # Row count per table
        (rows, _) = zip(*self.embedding_specs)
        # Cumulative row counts per table for rowwise states
        row_count_cumsum: List[int] = [0] + list(itertools.accumulate(rows))

        def copy_optimizer_state_(dst: Tensor, src: Tensor, indices: Tensor) -> None:
            device = dst.device
            dst.index_put_(
                indices=(
                    # indices is expected to be a tuple of Tensors, not Tensor
                    indices.to(device).view(-1),
                ),
                values=src.to(device),
            )

        for t, _ in enumerate(rows):
            # pyre-ignore [16]
            bucket_id_start, _ = self.kv_zch_params.bucket_offsets[t]
            # pyre-ignore [16]
            bucket_size = self.kv_zch_params.bucket_sizes[t]
            row_offset = row_count_cumsum[t] - bucket_id_start * bucket_size

            # pyre-ignore [16]
            weights = self._cached_kvzch_data.cached_weight_tensor_per_table[t]
            # pyre-ignore [16]
            ids = self._cached_kvzch_data.cached_id_tensor_per_table[t]
            local_ids = ids + row_offset

            logging.info(
                f"applying sd for table {t} without optimizer offloading, local_ids is {local_ids}"
            )
            # pyre-ignore [16]
            opt_states = self._cached_kvzch_data.cached_optimizer_states_per_table[t]

            # Set up the plan for copying optimizer states over
            if self.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
                mapping = [(opt_states[0], self.momentum1_dev)]
            elif self.optimizer == OptimType.PARTIAL_ROWWISE_ADAM:
                mapping = [
                    (opt_states[0], self.momentum1_dev),
                    (opt_states[1], self.momentum2_dev),
                ]
            else:
                mapping = []

            # Execute the plan and copy the optimizer states over
            # pyre-ignore [6]
            [copy_optimizer_state_(dst, src, local_ids) for (src, dst) in mapping]

            self.ssd_db.set_cuda(
                local_ids.view(-1),
                weights,
                torch.as_tensor(local_ids.size(0)),
                1,
                False,
            )

    @torch.jit.ignore
    def apply_state_dict(self) -> None:
        if self.backend_return_whole_row:
            logging.info(
                "backend_return_whole_row is enabled, no need to apply_state_dict"
            )
            return
        # After checkpoint loading, the _cached_kvzch_data will be loaded from checkpoint.
        # Caller should call this function to apply the cached states to backend.
        if self.load_state_dict is False:
            return
        self.load_state_dict = False
        assert self.kv_zch_params is not None, "apply_state_dict supports KV ZCH only"
        assert (
            self._cached_kvzch_data is not None
            and self._cached_kvzch_data.cached_optimizer_states_per_table is not None
        ), "optimizer state is not initialized for load checkpointing"
        assert (
            self._cached_kvzch_data.cached_weight_tensor_per_table is not None
            and self._cached_kvzch_data.cached_id_tensor_per_table is not None
        ), "weight and id state is not initialized for load checkpointing"

        # Compute the number of elements of cache_dtype needed to store the
        # optimizer state, round to the nearest 4
        # optimizer_dim = self.optimizer.optimizer_state_size_dim(dtype)
        # apply weight and optimizer state per table
        if self.enable_optimizer_offloading:
            self._apply_state_dict_w_offloading()
        else:
            self._apply_state_dict_no_offloading()

        self.clear_cache()

    @torch.jit.ignore
    def streaming_write_weight_and_id_per_table(
        self,
        weight_state: torch.Tensor,
        opt_states: List[torch.Tensor],
        id_tensor: torch.Tensor,
        row_offset: int,
    ) -> None:
        """
        This function is used to write weight, optimizer and id to the backend using kvt wrapper.
        to avoid over use memory, we will write the weight and id to backend in a rolling window manner

        Args:
            weight_state (torch.tensor): The weight state tensor to be written.
            opt_states (torch.tensor): The optimizer state tensor(s) to be written.
            id_tensor (torch.tensor): The id tensor to be written.
        """
        D = weight_state.size(1)
        dtype = self.weights_precision.as_dtype()

        optimizer_state_byte_offsets = self.optimizer.byte_offsets_along_row(
            D, self.weights_precision, self.optimizer_state_dtypes
        )
        optimizer_state_size_table = self.optimizer.state_size_table(D)

        kvt = torch.classes.fbgemm.KVTensorWrapper(
            shape=[weight_state.size(0), self.cache_row_dim],
            dtype=dtype,
            row_offset=row_offset,
            snapshot_handle=None,
            sorted_indices=id_tensor,
        )
        (
            kvt.set_embedding_rocks_dp_wrapper(self.ssd_db)
            if self.backend_type == BackendType.SSD
            else kvt.set_dram_db_wrapper(self.ssd_db)
        )

        # TODO: make chunk_size configurable or dynamic
        chunk_size = 10000
        row = weight_state.size(0)

        for i in range(0, row, chunk_size):
            # Construct the chunk buffer, using the weights precision as the dtype
            length = min(chunk_size, row - i)
            chunk_buffer = torch.empty(
                length,
                self.cache_row_dim,
                dtype=dtype,
                device="cpu",
            )

            # Copy the weight state over to the chunk buffer
            chunk_buffer[:, : weight_state.size(1)] = weight_state[i : i + length, :]

            # Copy the optimizer state(s) over to the chunk buffer
            for o, opt_state in enumerate(opt_states):
                # Fetch the state name based on the index
                state_name = self.optimizer.state_names()[o]

                # Fetch the byte offsets for the optimizer state by its name
                (start, end) = optimizer_state_byte_offsets[state_name]

                # Assume that the opt_state passed in already has dtype matching
                # self.optimizer_state_dtypes[state_name]
                opt_state_byteview = opt_state.view(
                    # Force it to be 2D table, with row size matching the
                    # optimizer state size
                    -1,
                    optimizer_state_size_table[state_name],
                ).view(
                    # Then force tensor to byte view
                    dtype=torch.uint8
                )

                # Convert the chunk buffer and optimizer state to byte views
                # Then use the start and end offsets to narrow the chunk buffer
                # and copy opt_state over
                chunk_buffer.view(dtype=torch.uint8)[:, start:end] = opt_state_byteview[
                    i : i + length, :
                ]

            # Write chunk to KVTensor
            kvt.set_weights_and_ids(chunk_buffer, id_tensor[i : i + length, :].view(-1))

    @torch.jit.ignore
    def enable_load_state_dict_mode(self) -> None:
        if self.backend_return_whole_row:
            logging.info(
                "backend_return_whole_row is enabled, no need to enable load_state_dict mode"
            )
            return
        # Enable load state dict mode before loading checkpoint
        if self.load_state_dict:
            return
        self.load_state_dict = True

        dtype = self.weights_precision.as_dtype()
        (_, dims) = zip(*self.embedding_specs)

        self._cached_kvzch_data = KVZCHCachedData([], [], [], [])

        for i, _ in enumerate(self.embedding_specs):
            # For checkpointing loading, we need to store the weight and id
            # tensor temporarily in memory.  First check that the local_weight_counts
            # are properly set before even initializing the optimizer states
            assert (
                self.local_weight_counts[i] > 0
            ), f"local_weight_counts for table {i} is not set"

        # pyre-ignore [16]
        self._cached_kvzch_data.cached_optimizer_states_per_table = (
            self.optimizer.empty_states(
                self.local_weight_counts,
                dims,
                self.optimizer_state_dtypes,
            )
        )

        for i, (_, emb_dim) in enumerate(self.embedding_specs):
            # pyre-ignore [16]
            bucket_id_start, bucket_id_end = self.kv_zch_params.bucket_offsets[i]
            rows = self.local_weight_counts[i]
            weight_state = torch.empty(rows, emb_dim, dtype=dtype, device="cpu")
            # pyre-ignore [16]
            self._cached_kvzch_data.cached_weight_tensor_per_table.append(weight_state)
            logging.info(
                f"for checkpoint loading, table {i}, weight_state shape is {weight_state.shape}"
            )
            id_tensor = torch.zeros((rows, 1), dtype=torch.int64, device="cpu")
            # pyre-ignore [16]
            self._cached_kvzch_data.cached_id_tensor_per_table.append(id_tensor)
            # pyre-ignore [16]
            self._cached_kvzch_data.cached_bucket_splits.append(
                torch.empty(
                    (bucket_id_end - bucket_id_start, 1),
                    dtype=torch.int64,
                    device="cpu",
                )
            )

    @torch.jit.export
    def set_learning_rate(self, lr: float) -> None:
        """
        Sets the learning rate.

        Args:
            lr (float): The learning rate value to set to
        """
        self._set_learning_rate(lr)

    def get_learning_rate(self) -> float:
        """
        Get and return the learning rate.
        """
        return self.learning_rate_tensor.item()

    @torch.jit.ignore
    def _set_learning_rate(self, lr: float) -> float:
        """
        Helper function to script `set_learning_rate`.
        Note that returning None does not work.
        """
        self.learning_rate_tensor = torch.tensor(
            lr, device=torch.device("cpu"), dtype=torch.float32
        )
        return 0.0

    def flush(self, force: bool = False) -> None:
        # allow force flush from split_embedding_weights to cover edge cases, e.g. checkpointing
        # after trained 0 batches
        if self.step == self.last_flush_step and not force:
            logging.info(
                f"SSD TBE has been flushed at {self.last_flush_step=} already for tbe:{self.tbe_unique_id}"
            )
            return
        logging.info(
            f"SSD TBE flush at {self.step=}, it is an expensive call please be cautious"
        )
        active_slots_mask = self.lxu_cache_state != -1

        active_weights_gpu = self.lxu_cache_weights[active_slots_mask.view(-1)].view(
            -1, self.cache_row_dim
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
        self.last_flush_step = self.step

    def create_rocksdb_hard_link_snapshot(self) -> None:
        """
        Create a rocksdb hard link snapshot to provide cross procs access to the underlying data
        """
        if self.backend_type == BackendType.SSD:
            self.ssd_db.create_rocksdb_hard_link_snapshot(self.step)
        else:
            logging.warning(
                "create_rocksdb_hard_link_snapshot is only supported for SSD backend"
            )

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
                bounds_check_version=self.bounds_check_version,
            )

        return indices, offsets, per_sample_weights, vbe_metadata

    @torch.jit.ignore
    def _report_kv_backend_stats(self) -> None:
        """
        All ssd stats report function entrance
        """
        if self.stats_reporter is None:
            return

        if not self.stats_reporter.should_report(self.step):
            return
        self._report_ssd_l1_cache_stats()

        if self.backend_type == BackendType.SSD:
            self._report_ssd_io_stats()
            self._report_ssd_mem_usage()
            self._report_l2_cache_perf_stats()
        if self.backend_type == BackendType.DRAM:
            self._report_dram_kv_perf_stats()
            if self.kv_zch_params and self.kv_zch_params.eviction_policy:
                self._report_eviction_stats()

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
                    * self.cache_row_dim
                    / passed_steps
                ),
            )

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

        # pyre-ignore [16]
        self.stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="ssd.io_duration.read_us",
            duration_ms=ssd_read_dur_us,
            time_unit="us",
        )

        self.stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="ssd.io_duration.write.fwd_rocksdb_read_us",
            duration_ms=fwd_rocksdb_read_dur,
            time_unit="us",
        )

        self.stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="ssd.io_duration.write.fwd_l1_eviction_us",
            duration_ms=fwd_l1_eviction_dur,
            time_unit="us",
        )

        self.stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="ssd.io_duration.write.bwd_l1_cnflct_miss_write_back_us",
            duration_ms=bwd_l1_cnflct_miss_write_back_dur,
            time_unit="us",
        )

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

        # pyre-ignore [16]
        self.stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name="ssd.mem_usage.block_cache",
            data_bytes=block_cache_usage,
        )

        self.stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name="ssd.mem_usage.estimate_table_reader",
            data_bytes=estimate_table_reader_usage,
        )

        self.stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name="ssd.mem_usage.memtable",
            data_bytes=memtable_usage,
        )

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

    @torch.jit.ignore
    def _report_eviction_stats(self) -> None:
        if self.stats_reporter is None:
            return

        stats_reporter: TBEStatsReporter = self.stats_reporter
        if not stats_reporter.should_report(self.step):
            return

        # skip metrics reporting when evicting disabled
        if self.kv_zch_params.eviction_policy.eviction_trigger_mode == 0:
            return

        T = len(set(self.feature_table_map))
        evicted_counts = torch.zeros(T, dtype=torch.int64)
        processed_counts = torch.zeros(T, dtype=torch.int64)
        full_duration_ms = torch.tensor(0, dtype=torch.int64)
        exec_duration_ms = torch.tensor(0, dtype=torch.int64)
        self.ssd_db.get_feature_evict_metric(
            evicted_counts, processed_counts, full_duration_ms, exec_duration_ms
        )

        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name=self.eviction_sum_evicted_counts_stats_name,
            data_bytes=int(evicted_counts.sum().item()),
        )
        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name=self.eviction_sum_processed_counts_stats_name,
            data_bytes=int(processed_counts.sum().item()),
        )
        if processed_counts.sum().item() != 0:
            stats_reporter.report_data_amount(
                iteration_step=self.step,
                event_name=self.eviction_evict_rate_stats_name,
                data_bytes=int(
                    evicted_counts.sum().item() * 100 / processed_counts.sum().item()
                ),
            )
        for t in self.feature_table_map:
            stats_reporter.report_data_amount(
                iteration_step=self.step,
                event_name=f"eviction.feature_table.{t}.evicted_counts",
                data_bytes=int(evicted_counts[t].item()),
            )
            stats_reporter.report_data_amount(
                iteration_step=self.step,
                event_name=f"eviction.feature_table.{t}.processed_counts",
                data_bytes=int(processed_counts[t].item()),
            )
            if processed_counts[t].item() != 0:
                stats_reporter.report_data_amount(
                    iteration_step=self.step,
                    event_name=f"eviction.feature_table.{t}.evict_rate",
                    data_bytes=int(
                        evicted_counts[t].item() * 100 / processed_counts[t].item()
                    ),
                )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="eviction.feature_table.full_duration_ms",
            duration_ms=full_duration_ms.item(),
            time_unit="ms",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="eviction.feature_table.exec_duration_ms",
            duration_ms=exec_duration_ms.item(),
            time_unit="ms",
        )
        if full_duration_ms.item() != 0:
            stats_reporter.report_data_amount(
                iteration_step=self.step,
                event_name="eviction.feature_table.exec_div_full_duration_rate",
                data_bytes=int(exec_duration_ms.item() * 100 / full_duration_ms.item()),
            )

    @torch.jit.ignore
    def _report_dram_kv_perf_stats(self) -> None:
        """
        EmbeddingKVDB will hold stats for DRAM cache performance in fwd/bwd
        this function fetch the stats from EmbeddingKVDB and report it with stats_reporter
        """
        if self.stats_reporter is None:
            return

        stats_reporter: TBEStatsReporter = self.stats_reporter
        if not stats_reporter.should_report(self.step):
            return

        dram_kv_perf_stats = self.ssd_db.get_dram_kv_perf(
            self.step, stats_reporter.report_interval  # pyre-ignore
        )

        if len(dram_kv_perf_stats) != 22:
            logging.error("dram cache perf stats should have 22 elements")
            return

        dram_read_duration = dram_kv_perf_stats[0]
        dram_read_sharding_duration = dram_kv_perf_stats[1]
        dram_read_cache_hit_copy_duration = dram_kv_perf_stats[2]
        dram_read_fill_row_storage_duration = dram_kv_perf_stats[3]
        dram_read_lookup_cache_duration = dram_kv_perf_stats[4]
        dram_read_acquire_lock_duration = dram_kv_perf_stats[5]
        dram_read_missing_load = dram_kv_perf_stats[6]
        dram_write_sharing_duration = dram_kv_perf_stats[7]

        dram_fwd_l1_eviction_write_duration = dram_kv_perf_stats[8]
        dram_fwd_l1_eviction_write_allocate_duration = dram_kv_perf_stats[9]
        dram_fwd_l1_eviction_write_cache_copy_duration = dram_kv_perf_stats[10]
        dram_fwd_l1_eviction_write_lookup_cache_duration = dram_kv_perf_stats[11]
        dram_fwd_l1_eviction_write_acquire_lock_duration = dram_kv_perf_stats[12]
        dram_fwd_l1_eviction_write_missing_load = dram_kv_perf_stats[13]

        dram_bwd_l1_cnflct_miss_write_duration = dram_kv_perf_stats[14]
        dram_bwd_l1_cnflct_miss_write_allocate_duration = dram_kv_perf_stats[15]
        dram_bwd_l1_cnflct_miss_write_cache_copy_duration = dram_kv_perf_stats[16]
        dram_bwd_l1_cnflct_miss_write_lookup_cache_duration = dram_kv_perf_stats[17]
        dram_bwd_l1_cnflct_miss_write_acquire_lock_duration = dram_kv_perf_stats[18]
        dram_bwd_l1_cnflct_miss_write_missing_load = dram_kv_perf_stats[19]

        dram_kv_allocated_bytes = dram_kv_perf_stats[20]
        dram_kv_actual_used_chunk_bytes = dram_kv_perf_stats[21]

        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.get.dram_read_duration_us",
            duration_ms=dram_read_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.get.dram_read_sharding_duration_us",
            duration_ms=dram_read_sharding_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.get.dram_read_cache_hit_copy_duration_us",
            duration_ms=dram_read_cache_hit_copy_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.get.dram_read_fill_row_storage_duration_us",
            duration_ms=dram_read_fill_row_storage_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.get.dram_read_lookup_cache_duration_us",
            duration_ms=dram_read_lookup_cache_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.get.dram_read_acquire_lock_duration_us",
            duration_ms=dram_read_acquire_lock_duration,
            time_unit="us",
        )
        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name="dram_kv.perf.get.dram_read_missing_load",
            data_bytes=dram_read_missing_load,
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_write_sharing_duration_us",
            duration_ms=dram_write_sharing_duration,
            time_unit="us",
        )

        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_fwd_l1_eviction_write_duration_us",
            duration_ms=dram_fwd_l1_eviction_write_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_fwd_l1_eviction_write_allocate_duration_us",
            duration_ms=dram_fwd_l1_eviction_write_allocate_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_fwd_l1_eviction_write_cache_copy_duration_us",
            duration_ms=dram_fwd_l1_eviction_write_cache_copy_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_fwd_l1_eviction_write_lookup_cache_duration_us",
            duration_ms=dram_fwd_l1_eviction_write_lookup_cache_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_fwd_l1_eviction_write_acquire_lock_duration_us",
            duration_ms=dram_fwd_l1_eviction_write_acquire_lock_duration,
            time_unit="us",
        )
        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_fwd_l1_eviction_write_missing_load",
            data_bytes=dram_fwd_l1_eviction_write_missing_load,
        )

        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_bwd_l1_cnflct_miss_write_duration_us",
            duration_ms=dram_bwd_l1_cnflct_miss_write_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_bwd_l1_cnflct_miss_write_allocate_duration_us",
            duration_ms=dram_bwd_l1_cnflct_miss_write_allocate_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_bwd_l1_cnflct_miss_write_cache_copy_duration_us",
            duration_ms=dram_bwd_l1_cnflct_miss_write_cache_copy_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_bwd_l1_cnflct_miss_write_lookup_cache_duration_us",
            duration_ms=dram_bwd_l1_cnflct_miss_write_lookup_cache_duration,
            time_unit="us",
        )
        stats_reporter.report_duration(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_bwd_l1_cnflct_miss_write_acquire_lock_duration_us",
            duration_ms=dram_bwd_l1_cnflct_miss_write_acquire_lock_duration,
            time_unit="us",
        )
        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name="dram_kv.perf.set.dram_bwd_l1_cnflct_miss_write_missing_load",
            data_bytes=dram_bwd_l1_cnflct_miss_write_missing_load,
        )

        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name=self.dram_kv_allocated_bytes_stats_name,
            data_bytes=dram_kv_allocated_bytes,
        )
        stats_reporter.report_data_amount(
            iteration_step=self.step,
            event_name=self.dram_kv_actual_used_chunk_bytes_stats_name,
            data_bytes=dram_kv_actual_used_chunk_bytes,
        )

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

    def fetch_from_l1_sp_w_row_ids(
        self, row_ids: torch.Tensor, only_get_optimizer_states: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Fetch the optimizer states and/or weights from L1 and SP for given linearized row_ids.
        @return: updated_weights/optimizer_states, mask of which rows are filled
        """
        if not self.enable_optimizer_offloading and only_get_optimizer_states:
            raise RuntimeError(
                "Optimizer states are not offloaded, while only_get_optimizer_states is True"
            )

        # NOTE: Remove this once there is support for fetching multiple
        # optimizer states in fetch_from_l1_sp_w_row_ids
        if only_get_optimizer_states and self.optimizer not in [
            OptimType.EXACT_ROWWISE_ADAGRAD,
            OptimType.PARTIAL_ROWWISE_ADAM,
        ]:
            raise RuntimeError(
                f"Fetching optimizer states using fetch_from_l1_sp_w_row_ids() is not yet supported for {self.optimizer}"
            )

        def split_results_by_opt_states(
            updated_weights: torch.Tensor, cache_location_mask: torch.Tensor
        ) -> Tuple[List[torch.Tensor], torch.Tensor]:
            if not only_get_optimizer_states:
                return [updated_weights], cache_location_mask
            # TODO: support mixed dimension case
            # currently only supports tables with the same max_D dimension
            opt_to_dim = self.optimizer.byte_offsets_along_row(
                self.max_D, self.weights_precision, self.optimizer_state_dtypes
            )
            updated_opt_states = []
            for opt_name, dim in opt_to_dim.items():
                opt_dtype = self.optimizer._extract_dtype(
                    self.optimizer_state_dtypes, opt_name
                )
                updated_opt_states.append(
                    updated_weights.view(dtype=torch.uint8)[:, dim[0] : dim[1]].view(
                        dtype=opt_dtype
                    )
                )
            return updated_opt_states, cache_location_mask

        with torch.no_grad():
            weights_dtype = self.weights_precision.as_dtype()
            step = self.step
            with record_function(f"## fetch_from_l1_{step}_{self.tbe_unique_id} ##"):
                lxu_cache_locations: torch.Tensor = torch.ops.fbgemm.lxu_cache_lookup(
                    row_ids,
                    self.lxu_cache_state,
                    self.total_hash_size,
                )
                updated_weights = torch.empty(
                    row_ids.numel(),
                    self.cache_row_dim,
                    device=self.current_device,
                    dtype=weights_dtype,
                )

                # D2D copy cache
                cache_location_mask = lxu_cache_locations >= 0
                torch.ops.fbgemm.masked_index_select(
                    updated_weights,
                    lxu_cache_locations,
                    self.lxu_cache_weights,
                    torch.tensor(
                        [row_ids.numel()],
                        device=self.current_device,
                        dtype=torch.int32,
                    ),
                )

            with record_function(f"## fetch_from_sp_{step}_{self.tbe_unique_id} ##"):
                if len(self.ssd_scratch_pad_eviction_data) > 0:
                    sp = self.ssd_scratch_pad_eviction_data[0][0]
                    sp_idx = self.ssd_scratch_pad_eviction_data[0][1].to(
                        self.current_device
                    )
                    actions_count_gpu = self.ssd_scratch_pad_eviction_data[0][2][0]
                    if actions_count_gpu.item() == 0:
                        # no action to take
                        return split_results_by_opt_states(
                            updated_weights, cache_location_mask
                        )

                    sp_idx = sp_idx[:actions_count_gpu]

                    # -1 in lxu_cache_locations means the row is not in L1 cache and in SP
                    # fill the row_ids in L1 with -2, >0 values means in SP
                    # @eg. updated_row_ids_in_sp= [1, 100, 1, 2, -2, 3, 4, 5, 10]
                    updated_row_ids_in_sp = row_ids.masked_fill(
                        lxu_cache_locations != -1, -2
                    )
                    # sort the sp_idx for binary search
                    # should already be sorted
                    # sp_idx_inverse_indices is the indices before sorting which is same to the location in SP.
                    # @eg. sp_idx = [4, 2, 1, 3, 10]
                    # @eg sorted_sp_idx = [ 1,  2,  3,  4, 10] and sp_idx_inverse_indices = [2, 1, 3, 0, 4]
                    sorted_sp_idx, sp_idx_inverse_indices = torch.sort(sp_idx)
                    # search rows id in sp against the SP indexes to find location of the rows in SP
                    # @eg: updated_ids_in_sp_idx = [0, 5, 0, 1, 0, 2, 3, 4, 4]
                    # @eg: 5 is OOB
                    updated_ids_in_sp_idx = torch.searchsorted(
                        sorted_sp_idx, updated_row_ids_in_sp
                    )
                    # does not found in SP will Out of Bound
                    oob_sp_idx = updated_ids_in_sp_idx >= sp_idx.numel()
                    # make the oob items in bound
                    # @eg updated_ids_in_sp_idx=[0, 0, 0, 1, 0, 2, 3, 4, 4]
                    updated_ids_in_sp_idx[oob_sp_idx] = 0

                    # -1s locations will be filtered out in masked_index_select
                    sp_locations_in_updated_weights = torch.full_like(
                        updated_row_ids_in_sp, -1
                    )
                    # torch.searchsorted is not exact match,
                    # we only take exact matched rows, where the id is found in SP.
                    # @eg 5 in updated_row_ids_in_sp is not in sp_idx, but has 4 in updated_ids_in_sp_idx
                    # @eg sorted_sp_idx[updated_ids_in_sp_idx]=[ 1,  1,  1,  2,  1,  3,  4, 10, 10]
                    # @eg exact_match_mask=[ True, False,  True,  True, False,  True,  True, False,  True]
                    exact_match_mask = (
                        sorted_sp_idx[updated_ids_in_sp_idx] == updated_row_ids_in_sp
                    )
                    # Get the location of the row ids found in SP.
                    # @eg: sp_locations_found=[2, 2, 1, 3, 0, 4]
                    sp_locations_found = sp_idx_inverse_indices[
                        updated_ids_in_sp_idx[exact_match_mask]
                    ]
                    # @eg: sp_locations_in_updated_weights=[ 2, -1,  2,  1, -1,  3,  0, -1,  4]
                    sp_locations_in_updated_weights[exact_match_mask] = (
                        sp_locations_found
                    )

                    # D2D copy SP
                    torch.ops.fbgemm.masked_index_select(
                        updated_weights,
                        sp_locations_in_updated_weights,
                        sp,
                        torch.tensor(
                            [row_ids.numel()],
                            device=self.current_device,
                            dtype=torch.int32,
                        ),
                    )
                    # cache_location_mask is the mask of rows in L1
                    # exact_match_mask is the mask of rows in SP
                    cache_location_mask = torch.logical_or(
                        cache_location_mask, exact_match_mask
                    )

            return split_results_by_opt_states(updated_weights, cache_location_mask)

    def register_backward_hook_before_eviction(
        self, backward_hook: Callable[[torch.Tensor], None]
    ) -> None:
        """
        Register a backward hook to the TBE module.
        And make sure this is called before the sp eviction hook.
        """
        # make sure this hook is the first one to be executed
        hooks = []
        backward_hooks = self.placeholder_autograd_tensor._backward_hooks
        if backward_hooks is not None:
            for _handle_id, hook in backward_hooks.items():
                hooks.append(hook)
            backward_hooks.clear()

        self.placeholder_autograd_tensor.register_hook(backward_hook)
        for hook in hooks:
            self.placeholder_autograd_tensor.register_hook(hook)
