#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import logging
from itertools import accumulate
from typing import List, Optional, Tuple, Union

import torch  # usort:skip
from torch import nn, Tensor  # usort:skip

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheState,
    construct_cache_state,
    DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    EmbeddingLocation,
    MAX_PREFETCH_DEPTH,
    PoolingMode,
    RecordCacheMetrics,
    round_up,
    SplitState,
)

try:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cuda_inference"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cpu_inference"
    )
except Exception:
    pass


def rounded_row_size_in_bytes(
    dim: int,
    weight_ty: SparseType,
    row_alignment: int,
    scale_bias_size_in_bytes: int = DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
) -> int:
    r = unpadded_row_size_in_bytes(dim, weight_ty, scale_bias_size_in_bytes)
    # align each row to 16-byte boundaries.
    return round_up(r, row_alignment)


def unpadded_row_size_in_bytes(
    dim: int,
    weight_ty: SparseType,
    scale_bias_size_in_bytes: int = DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
) -> int:
    r = {
        SparseType.FP32.value: dim * 4,
        SparseType.FP16.value: dim * 2,
        SparseType.FP8.value: dim,
        SparseType.INT8.value: dim + scale_bias_size_in_bytes,
        SparseType.INT4.value: dim // 2 + scale_bias_size_in_bytes,
        SparseType.INT2.value: dim // 4 + scale_bias_size_in_bytes,
    }[weight_ty.value]
    return r


def align_to_cacheline(a: int) -> int:
    # align each table to 128b cache line boundary.
    return round_up(a, 128)


def nbit_construct_split_state(
    embedding_specs: List[Tuple[str, int, int, SparseType, EmbeddingLocation]],
    cacheable: bool,
    row_alignment: int,
    scale_bias_size_in_bytes: int = DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    cacheline_alignment: bool = True,
) -> SplitState:
    placements = torch.jit.annotate(List[EmbeddingLocation], [])
    offsets = torch.jit.annotate(List[int], [])
    dev_size = 0
    host_size = 0
    uvm_size = 0
    for _, num_embeddings, embedding_dim, weight_ty, location in embedding_specs:
        embedding_dim = rounded_row_size_in_bytes(
            embedding_dim, weight_ty, row_alignment, scale_bias_size_in_bytes
        )
        state_size = num_embeddings * embedding_dim
        if cacheline_alignment:
            state_size = align_to_cacheline(state_size)
        if location == EmbeddingLocation.HOST:
            placements.append(EmbeddingLocation.HOST)
            offsets.append(host_size)
            host_size += state_size
        elif location == EmbeddingLocation.DEVICE:
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


def random_quant_scaled_tensor(shape: torch.Size, device: torch.device) -> torch.Tensor:
    return torch.randint(
        0,
        255,
        size=shape,
        dtype=torch.uint8,
        device=device,
    )


# pyre-fixme[13]: Attribute `cache_miss_counter` is never initialized.
class IntNBitTableBatchedEmbeddingBagsCodegen(nn.Module):
    """
    Table-batched version of nn.EmbeddingBag(sparse=False)
    Inference version, with FP32/FP16/FP8/INT8/INT4/INT2 supports
    """

    embedding_specs: List[Tuple[str, int, int, SparseType, EmbeddingLocation]]
    record_cache_metrics: RecordCacheMetrics
    cache_miss_counter: torch.Tensor
    uvm_cache_stats: torch.Tensor
    local_uvm_cache_stats: torch.Tensor
    weights_offsets: torch.Tensor
    weights_placements: torch.Tensor

    def __init__(  # noqa C901
        self,
        embedding_specs: List[
            Tuple[str, int, int, SparseType, EmbeddingLocation]
        ],  # tuple of (feature_names, rows, dims, SparseType, EmbeddingLocation/placement)
        feature_table_map: Optional[List[int]] = None,  # [T]
        index_remapping: Optional[List[Tensor]] = None,
        pooling_mode: PoolingMode = PoolingMode.SUM,
        device: Optional[Union[str, int, torch.device]] = None,
        bounds_check_mode: BoundsCheckMode = BoundsCheckMode.WARNING,
        weight_lists: Optional[List[Tuple[Tensor, Optional[Tensor]]]] = None,
        pruning_hash_load_factor: float = 0.5,
        use_array_for_index_remapping: bool = True,
        output_dtype: SparseType = SparseType.FP16,
        cache_algorithm: CacheAlgorithm = CacheAlgorithm.LRU,
        cache_load_factor: float = 0.2,
        cache_sets: int = 0,
        cache_reserved_memory: float = 0.0,
        enforce_hbm: bool = False,  # place all weights/momentums in HBM when using cache
        record_cache_metrics: Optional[RecordCacheMetrics] = None,
        gather_uvm_cache_stats: Optional[bool] = False,
        row_alignment: Optional[int] = None,
        fp8_exponent_bits: Optional[int] = None,
        fp8_exponent_bias: Optional[int] = None,
        cache_assoc: int = 32,
        scale_bias_size_in_bytes: int = DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
        cacheline_alignment: bool = True,
        uvm_host_mapped: bool = False,  # True to use cudaHostAlloc; False to use cudaMallocManaged.
        reverse_qparam: bool = False,  # True to load qparams at end of each row; False to load qparam at begnning of each row.
    ) -> None:  # noqa C901  # tuple of (rows, dims,)
        super(IntNBitTableBatchedEmbeddingBagsCodegen, self).__init__()

        # 64 for AMD
        if cache_assoc == 32 and torch.version.hip is not None:
            cache_assoc = 64

        if device is None:
            self.current_device: torch.device = torch.device(
                torch.cuda.current_device()
            )
        elif isinstance(device, torch.device):
            self.current_device = device
        else:
            self.current_device = torch.device(device)
        self.use_cpu: bool = self.current_device.type == "cpu"

        self.scale_bias_size_in_bytes = scale_bias_size_in_bytes
        self.pooling_mode = pooling_mode
        self.bounds_check_mode_int: int = bounds_check_mode.value
        self.embedding_specs = embedding_specs
        self.output_dtype: int = output_dtype.as_int()
        self.uvm_host_mapped = uvm_host_mapped
        # (feature_names, rows, dims, weights_tys, locations) = zip(*embedding_specs)
        # Pyre workaround
        self.feature_names: List[str] = [e[0] for e in embedding_specs]
        rows: List[int] = [e[1] for e in embedding_specs]
        dims: List[int] = [e[2] for e in embedding_specs]
        weights_tys: List[SparseType] = [e[3] for e in embedding_specs]
        locations: List[EmbeddingLocation] = [e[4] for e in embedding_specs]
        # if target device is meta then we set use_cpu based on the embedding location
        # information in embedding_specs.
        if self.current_device.type == "meta":
            self.use_cpu = all(loc == EmbeddingLocation.HOST for loc in locations)

        if row_alignment is None:
            self.row_alignment: int = 1 if self.use_cpu else 16
        else:
            self.row_alignment = row_alignment

        if record_cache_metrics is not None:
            self.record_cache_metrics = record_cache_metrics
        else:
            self.record_cache_metrics = RecordCacheMetrics(False, False)

        self.gather_uvm_cache_stats = gather_uvm_cache_stats
        # Define the size of uvm cache stats as class variable
        # to make it work with torch jit script.
        self.uvm_cache_stats_size = 6
        # 0: N_calls, 1: N_requested_indices, 2: N_unique_indices, 3: N_unique_misses,
        # 4: N_conflict_unique_misses, 5: N_conflict_misses

        # mixed D is not supported by no bag kernels
        mixed_D = not all(d == dims[0] for d in dims)
        if mixed_D:
            assert (
                self.pooling_mode != PoolingMode.NONE
            ), "Mixed dimension tables are only supported for pooling tables."

        assert not self.use_cpu or all(
            loc == EmbeddingLocation.HOST for loc in locations
        ), "CPU device requires EmbeddingLocation.HOST for location!"
        assert self.use_cpu or all(
            loc != EmbeddingLocation.HOST for loc in locations
        ), "EmbeddingLocation.HOST doesn't work for CUDA device!"

        T_ = len(self.embedding_specs)
        assert T_ > 0

        self.feature_table_map: List[int] = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        T = len(self.feature_table_map)
        assert T_ <= T

        table_has_feature = [False] * T_
        for t in self.feature_table_map:
            table_has_feature[t] = True
        assert all(table_has_feature), "Each table must have at least one feature!"
        D_offsets = [dims[t] for t in self.feature_table_map]
        D_offsets = [0] + list(accumulate(D_offsets))
        self.total_D: int = D_offsets[-1]
        for dim, weight_ty in zip(dims, weights_tys):
            if not weight_ty.is_float():
                assert (
                    dim % (8 / weight_ty.bit_rate()) == 0
                ), f"For quantized types we need to at least pack at byte granularity, dim: {dim}, weight_ty: {weight_ty}"

        def max_ty_D(ty: SparseType) -> int:
            return max(
                [dim for dim, weight_ty in zip(dims, weights_tys) if weight_ty == ty],
                default=0,
            )

        self.max_int2_D: int = max_ty_D(SparseType.INT2)
        self.max_int4_D: int = max_ty_D(SparseType.INT4)
        self.max_int8_D: int = max_ty_D(SparseType.INT8)
        self.max_float8_D: int = max_ty_D(SparseType.FP8)
        self.max_float16_D: int = max_ty_D(SparseType.FP16)
        self.max_float32_D: int = max_ty_D(SparseType.FP32)

        self.register_buffer(
            "D_offsets",
            torch.tensor(D_offsets, device=self.current_device, dtype=torch.int32),
        )
        assert self.D_offsets.numel() == T + 1

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

        weights_tys_int = [weights_tys[t].as_int() for t in self.feature_table_map]
        self.register_buffer(
            "weights_tys",
            torch.tensor(
                weights_tys_int, device=self.current_device, dtype=torch.uint8
            ),
        )
        self.weight_initialized: bool = False

        self.weights_dev: torch.Tensor = torch.zeros(
            0,
            device=self.current_device,
            dtype=torch.uint8,
        )

        self.weights_host: torch.Tensor = torch.zeros(
            0, device=self.current_device, dtype=torch.uint8
        )

        self.weights_uvm: torch.Tensor = torch.empty(0, dtype=torch.uint8).to(
            self.current_device
        )

        cached_dims = [
            rounded_row_size_in_bytes(
                embedding_spec[2], embedding_spec[3], 16, self.scale_bias_size_in_bytes
            )
            for embedding_spec in self.embedding_specs
            if embedding_spec[4] == EmbeddingLocation.MANAGED_CACHING
        ]
        self.max_D_cache: int = max(cached_dims) if len(cached_dims) > 0 else 0

        self.initialize_physical_weights_placements_and_offsets(cacheline_alignment)
        self.enforce_hbm: bool = enforce_hbm

        self.reverse_qparam = reverse_qparam
        # Assign weights after weights and weights_offsets are initialized.
        if weight_lists:
            self._apply_split(
                self.dev_size,
                self.host_size,
                self.uvm_size,
                self.weights_physical_placements,
                self.weights_physical_offsets,
                self.enforce_hbm,
            )
            self.assign_embedding_weights(weight_lists)

        # Handle index remapping for embedding pruning.
        self.register_buffer(
            "index_remappings_array_offsets",
            torch.empty(0, device=self.current_device, dtype=torch.int64),
        )
        self.register_buffer(
            "index_remappings_array",
            torch.empty(0, device=self.current_device, dtype=torch.int32),
        )
        self.register_buffer(
            "index_remapping_hash_table_offsets",
            torch.empty(0, device=self.current_device, dtype=torch.int64),
        )
        self.register_buffer(
            "index_remapping_hash_table",
            torch.empty(0, device=self.current_device, dtype=torch.int32),
        )
        self.register_buffer(
            "original_rows_per_table",
            torch.empty(0, device=self.current_device, dtype=torch.int64),
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.index_remapping_hash_table_cpu = None

        if index_remapping:
            self.set_index_remappings(
                index_remapping, pruning_hash_load_factor, use_array_for_index_remapping
            )

        # Currently only support cache_precision == embedding_precision.
        # Both are represented as uint8_t
        cache_state = construct_cache_state(rows, locations, self.feature_table_map)

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

        self.cache_assoc = cache_assoc
        self._apply_cache_state(
            cache_state,
            cache_algorithm,
            cache_load_factor,
            cache_sets,
            cache_reserved_memory,
        )

        if self.max_float8_D > 0:
            default_config = SparseType.FP8.default_config()
            self.fp8_exponent_bits: int = (
                default_config.get("exponent_bits")
                if fp8_exponent_bits is None
                else fp8_exponent_bits
            )
            self.fp8_exponent_bias: int = (
                default_config.get("exponent_bias")
                if fp8_exponent_bias is None
                else fp8_exponent_bias
            )
        else:
            self.fp8_exponent_bits = -1
            self.fp8_exponent_bias = -1

    def get_cache_miss_counter(self) -> Tensor:
        # cache_miss_counter[0]: cache_miss_forward_count which records the total number of forwards which has at least one cache miss
        # cache_miss_counter[1]: unique_cache_miss_count which records to total number of unique (dedup) cache misses
        # cache_miss_counter[2]: total number of unique (dedup) access count
        # cache_miss_counter[3]: total number of non-dedup access count

        # How to get cache miss ratio
        # cache miss ratio (# of missed entries / # of unique requests): ( cache_miss_counter[1] / cache_miss_counter[2] )
        # cache miss ratio (# of missed entries / # of total access): ( cache_miss_counter[1] / cache_miss_counter[3] )
        assert (
            self.record_cache_metrics.record_cache_miss_counter
        ), "record_cache_miss_counter should be true to access counter values"

        return self.cache_miss_counter

    @torch.jit.export
    def get_table_wise_cache_miss(self) -> Tensor:
        assert (
            self.record_cache_metrics.record_tablewise_cache_miss
        ), "record_tablewise_cache_miss should be true to access counter values"
        # table_wise_cache_miss contains all the cache miss count for each table in this embedding table object:
        return self.table_wise_cache_miss

    def reset_cache_miss_counter(self) -> None:
        assert (
            self.record_cache_metrics.record_cache_miss_counter
        ), "record_cache_miss_counter should be true to access counter values"
        self.cache_miss_counter = torch.tensor(
            [0, 0, 0, 0], device=self.current_device, dtype=torch.int64
        )

    def reset_uvm_cache_stats(self) -> None:
        assert (
            self.gather_uvm_cache_stats
        ), "gather_uvm_cache_stats should be set to true to access uvm cache stats."
        self.uvm_cache_stats.zero_()
        self.local_uvm_cache_stats.zero_()

    def print_cache_miss_counter(self) -> None:
        assert (
            self.record_cache_metrics.record_cache_miss_counter
        ), "record_cache_miss_counter should be true to access counter values"
        logging.info(
            f"\n"
            f"Miss counter value [0] - # of miss occured iters : {self.cache_miss_counter[0]}, \n"
            f"Miss counter value [1] - # of unique misses : {self.cache_miss_counter[1]}, \n"
            f"Miss counter value [2] - # of unique requested indices : {self.cache_miss_counter[2]}, \n"
            f"Miss counter value [3] - # of total requested indices : {self.cache_miss_counter[3]}, "
        )
        logging.info(
            f"unique_miss_rate using counter : {self.cache_miss_counter[1]/self.cache_miss_counter[2]}, \n"
        )
        logging.info(
            f"total_miss_rate using counter : {self.cache_miss_counter[1]/self.cache_miss_counter[3]}, \n"
        )

    def get_uvm_cache_stats(self) -> Tensor:
        assert (
            self.gather_uvm_cache_stats
        ), "gather_uvm_cache_stats should be set to true to access uvm cache stats."
        return self.uvm_cache_stats

    def print_uvm_cache_stats(self) -> None:
        assert (
            self.gather_uvm_cache_stats
        ), "gather_uvm_cache_stats should be set to true to access uvm cache stats."
        uvm_cache_stats = self.uvm_cache_stats.tolist()
        logging.info(
            f"N_called: {uvm_cache_stats[0]}\n"
            f"N_requested_indices: {uvm_cache_stats[1]}\n"
            f"N_unique_indices: {uvm_cache_stats[2]}\n"
            f"N_unique_misses: {uvm_cache_stats[3]}\n"
            f"N_conflict_unique_misses: {uvm_cache_stats[4]}\n"
            f"N_conflict_misses: {uvm_cache_stats[5]}\n"
        )
        if uvm_cache_stats[1]:
            logging.info(
                f"unique indices / requested indices: {uvm_cache_stats[2]/uvm_cache_stats[1]}\n"
                f"unique misses / requested indices: {uvm_cache_stats[3]/uvm_cache_stats[1]}\n"
            )

    @torch.jit.export
    def prefetch(self, indices: Tensor, offsets: Tensor) -> None:
        self.timestep_counter.increment()
        self.timestep_prefetch_size.increment()
        if not self.lxu_cache_weights.numel():
            return

        linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
            self.cache_hash_size_cumsum,
            indices,
            offsets,
        )

        if (
            self.record_cache_metrics.record_cache_miss_counter
            or self.record_cache_metrics.record_tablewise_cache_miss
        ):
            lxu_cache_locations = (
                torch.ops.fbgemm.lxu_cache_lookup(
                    linear_cache_indices,
                    self.lxu_cache_state,
                    self.total_cache_hash_size,
                )
                if self.cache_assoc in [32, 64]
                else torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(
                    linear_cache_indices,
                    self.lxu_cache_state,
                    self.total_cache_hash_size,
                )
            )
            if self.record_cache_metrics.record_cache_miss_counter:
                self._update_cache_miss_counter(
                    lxu_cache_locations, linear_cache_indices
                )
            if self.record_cache_metrics.record_tablewise_cache_miss:
                self._update_tablewise_cache_miss(
                    lxu_cache_locations, linear_cache_indices, offsets
                )

        if self.cache_assoc in [32, 64]:
            # 64 for AMD
            self.prefetch_32way(linear_cache_indices)
        elif self.cache_assoc == 1:
            self.prefetch_1way(linear_cache_indices)
        else:
            raise ValueError(f"{self.cache_assoc} not in [1, 32, 64]")

    def prefetch_32way(self, linear_cache_indices: Tensor) -> None:
        if self.cache_algorithm == CacheAlgorithm.LRU:
            torch.ops.fbgemm.lru_cache_populate_byte(
                self.weights_uvm,
                self.cache_hash_size_cumsum,
                self.total_cache_hash_size,
                self.cache_index_table_map,
                self.weights_offsets,
                self.weights_tys,
                self.D_offsets,
                linear_cache_indices,
                self.lxu_cache_state,
                self.lxu_cache_weights,
                self.timestep_counter.get(),
                self.lxu_state,
                16,  # row_alignment; using default value.
                self.gather_uvm_cache_stats,
                self.local_uvm_cache_stats,
            )
        elif self.cache_algorithm == CacheAlgorithm.LFU:
            torch.ops.fbgemm.lfu_cache_populate_byte(
                self.weights_uvm,
                self.cache_hash_size_cumsum,
                self.total_cache_hash_size,
                self.cache_index_table_map,
                self.weights_offsets,
                self.weights_tys,
                self.D_offsets,
                linear_cache_indices,
                self.lxu_cache_state,
                self.lxu_cache_weights,
                self.lxu_state,
            )

        assert (
            self.lxu_cache_locations_list.size() < self.max_prefetch_depth
        ), f"self.lxu_cache_locations_list has grown to size: {self.lxu_cache_locations_list.size()}, this exceeds the maximum: {self.max_prefetch_depth}. This probably indicates an error in logic where prefetch() is being called more frequently than forward()"
        self.lxu_cache_locations_list.push(
            torch.ops.fbgemm.lxu_cache_lookup(
                linear_cache_indices,
                self.lxu_cache_state,
                self.total_cache_hash_size,
                self.gather_uvm_cache_stats,
                self.local_uvm_cache_stats,
            )
        )
        if self.gather_uvm_cache_stats:
            self._accumulate_uvm_cache_stats()

    def prefetch_1way(self, linear_cache_indices: Tensor) -> None:
        if self.cache_algorithm == CacheAlgorithm.LRU:
            torch.ops.fbgemm.direct_mapped_lru_cache_populate_byte(
                self.weights_uvm,
                self.cache_hash_size_cumsum,
                self.total_cache_hash_size,
                self.cache_index_table_map,
                self.weights_offsets,
                self.weights_tys,
                self.D_offsets,
                linear_cache_indices,
                self.lxu_cache_state,
                self.lxu_cache_weights,
                self.timestep_counter.get(),
                self.lxu_state,
                self.lxu_cache_miss_timestamp,
                16,  # row_alignment; using default value.
                self.gather_uvm_cache_stats,
                self.local_uvm_cache_stats,
            )
        else:
            raise ValueError("Direct Mapped for LRU only")

        assert (
            self.lxu_cache_locations_list.size() < self.max_prefetch_depth
        ), f"self.lxu_cache_locations_list has grown to size: {self.lxu_cache_locations_list.size()}, this exceeds the maximum: {self.max_prefetch_depth}. This probably indicates an error in logic where prefetch() is being called more frequently than forward()"
        self.lxu_cache_locations_list.push(
            torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(
                linear_cache_indices,
                self.lxu_cache_state,
                self.total_cache_hash_size,
                self.gather_uvm_cache_stats,
                self.local_uvm_cache_stats,
            )
        )
        if self.gather_uvm_cache_stats:
            self._accumulate_uvm_cache_stats()

    def _accumulate_uvm_cache_stats(self) -> None:
        # Accumulate local_uvm_cache_stats (int32) into uvm_cache_stats (int64).
        # We may wanna do this accumulation atomically, but as it's only for monitoring,
        # slightly inaccurate result may be acceptable.
        self.uvm_cache_stats = torch.add(
            self.uvm_cache_stats, self.local_uvm_cache_stats
        )
        self.local_uvm_cache_stats.zero_()

    def _update_cache_miss_counter(
        self,
        lxu_cache_locations: Tensor,
        linear_cache_indices: Tensor,
    ) -> None:
        CACHE_MISS = torch.tensor([-1], device=self.current_device, dtype=torch.int32)
        CACHE_HIT = torch.tensor([-2], device=self.current_device, dtype=torch.int32)

        cache_missed_locations = torch.where(
            lxu_cache_locations == CACHE_MISS, linear_cache_indices, CACHE_HIT
        )
        unique_ids_list = torch.unique(cache_missed_locations)
        unique_ids_count_list = torch.where(unique_ids_list == CACHE_HIT, 0, 1)

        miss_count = torch.sum(unique_ids_count_list)

        self.cache_miss_counter[0] += (miss_count > 0).to(torch.int64)

        self.cache_miss_counter[1] += miss_count

        # Number of unique requests
        assert (
            len(linear_cache_indices.size()) == 1
        ), f"linear_cache_indices should be 1-D was {len(linear_cache_indices.size())}-D"

        assert (
            self.cache_miss_counter.size()[0] == 4
        ), f"self.cache_miss_counter should be 4-D was {self.cache_miss_counter.size()[0]}-D"

        self.cache_miss_counter[2] += torch.unique(linear_cache_indices).size()[0]

        # Number of total requests
        self.cache_miss_counter[3] += linear_cache_indices.size()[0]

    def _update_tablewise_cache_miss(
        self,
        lxu_cache_locations: Tensor,
        linear_cache_indices: Tensor,
        offsets: Tensor,
    ) -> None:
        CACHE_MISS = torch.tensor([-1], device=self.current_device, dtype=torch.int32)
        CACHE_HIT = torch.tensor([-2], device=self.current_device, dtype=torch.int32)

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

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        assert (
            self.weight_initialized
        ), "weight needs to be initialized before forward function"

        # First bound check: check if the indices/offsets are within the boundary
        # of the original embedding rows before pruning.
        # Note that this is only applied when we enable pruning (if the perf becomes
        # an issue, we can fuse it inside the remapping kernel).
        if (
            self.index_remapping_hash_table_cpu is not None
            or self.index_remapping_hash_table.numel() > 0
            or self.index_remappings_array.numel() > 0
        ):
            if self.bounds_check_mode_int != BoundsCheckMode.NONE.value:
                torch.ops.fbgemm.bounds_check_indices(
                    self.original_rows_per_table,
                    indices,
                    offsets,
                    self.bounds_check_mode_int,
                    self.bounds_check_warning,
                    per_sample_weights,
                )

        # Index remapping changes input indices, and some of them becomes -1 (prunned rows).
        # Hence, remapping should be done before prefetch and emb lookup
        # so that these operations are with the remapped indices.
        if self.index_remapping_hash_table_cpu is not None:
            indices = self.index_remapping_hash_table_cpu.lookup(indices, offsets)
        elif self.index_remapping_hash_table.numel() > 0:
            # Convert from raw indices to pruned indices
            indices = torch.ops.fbgemm.pruned_hashmap_lookup(
                indices,
                offsets,
                self.index_remapping_hash_table,
                self.index_remapping_hash_table_offsets,
            )
        elif self.index_remappings_array.numel() > 0:
            indices = torch.ops.fbgemm.pruned_array_lookup(
                indices,
                offsets,
                self.index_remappings_array,
                self.index_remappings_array_offsets,
            )
        if self.timestep_prefetch_size.get() <= 0:
            self.prefetch(indices, offsets)
        self.timestep_prefetch_size.decrement()

        lxu_cache_locations = self.lxu_cache_locations_list.pop()

        # Second bound check: check if the indices/offsets are within the boundary
        # of the pruned embedding rows after pruning.
        # Note: we cast to int as a TorchScript workaround.
        if self.bounds_check_mode_int != BoundsCheckMode.NONE.value:
            torch.ops.fbgemm.bounds_check_indices(
                self.rows_per_table,
                indices,
                offsets,
                self.bounds_check_mode_int,
                self.bounds_check_warning,
                per_sample_weights,
            )
        # Note: CPU and CUDA ops use the same interface to facilitate JIT IR
        # generation for CUDA/CPU. For CPU op, we don't need weights_uvm and
        # weights_placements
        return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
            dev_weights=self.weights_host if self.host_size > 0 else self.weights_dev,
            uvm_weights=self.weights_uvm,
            weights_placements=self.weights_placements,
            weights_offsets=self.weights_offsets,
            weights_tys=self.weights_tys,
            D_offsets=self.D_offsets,
            total_D=self.total_D,
            max_int2_D=self.max_int2_D,
            max_int4_D=self.max_int4_D,
            max_int8_D=self.max_int8_D,
            max_float16_D=self.max_float16_D,
            max_float32_D=self.max_float32_D,
            indices=indices,
            offsets=offsets,
            pooling_mode=int(self.pooling_mode),
            indice_weights=per_sample_weights,
            output_dtype=self.output_dtype,
            lxu_cache_weights=self.lxu_cache_weights,
            lxu_cache_locations=lxu_cache_locations,
            row_alignment=self.row_alignment,
            max_float8_D=self.max_float8_D,
            fp8_exponent_bits=self.fp8_exponent_bits,
            fp8_exponent_bias=self.fp8_exponent_bias,
        )

    def initialize_logical_weights_placements_and_offsets(
        self,
    ) -> None:
        assert len(self.weights_physical_offsets) == len(self.embedding_specs)
        assert len(self.weights_physical_offsets) == len(
            self.weights_physical_placements
        )
        offsets = [self.weights_physical_offsets[t] for t in self.feature_table_map]
        placements = [
            self.weights_physical_placements[t] for t in self.feature_table_map
        ]
        self.weights_offsets = torch.tensor(
            offsets, device=self.current_device, dtype=torch.int64
        )
        self.weights_placements = torch.tensor(
            placements, device=self.current_device, dtype=torch.int32
        )

    def initialize_physical_weights_placements_and_offsets(
        self,
        cacheline_alignment: bool = True,
    ) -> None:
        # Initialize physical weights placements and offsets
        # and host/dev/uvm sizes
        weight_split: SplitState = nbit_construct_split_state(
            self.embedding_specs,
            cacheable=True,
            row_alignment=self.row_alignment,
            scale_bias_size_in_bytes=self.scale_bias_size_in_bytes,
            cacheline_alignment=cacheline_alignment,
        )
        self.weights_physical_placements = [t.value for t in weight_split.placements]
        self.weights_physical_offsets = weight_split.offsets
        self.host_size = weight_split.host_size
        self.dev_size = weight_split.dev_size
        self.uvm_size = weight_split.uvm_size

    @torch.jit.export
    def reset_weights_placements_and_offsets(
        self, device: torch.device, location: int
    ) -> None:
        # Reset device/location denoted in embedding specs
        self.reset_embedding_spec_location(device, location)
        # Initialize all physical/logical weights placements and offsets without initializing large dev weights tensor
        self.initialize_physical_weights_placements_and_offsets()
        self.initialize_logical_weights_placements_and_offsets()

    def reset_embedding_spec_location(
        self, device: torch.device, location: int
    ) -> None:
        # Overwrite location in embedding_specs with new location
        # Use map since can't script enum call (ie. EmbeddingLocation(value))
        INT_TO_EMBEDDING_LOCATION = {
            0: EmbeddingLocation.DEVICE,
            1: EmbeddingLocation.MANAGED,
            2: EmbeddingLocation.MANAGED_CACHING,
            3: EmbeddingLocation.HOST,
        }
        target_location = INT_TO_EMBEDDING_LOCATION[location]
        self.current_device = device
        self.row_alignment = 1 if target_location == EmbeddingLocation.HOST else 16
        self.embedding_specs = [
            (spec[0], spec[1], spec[2], spec[3], target_location)
            for spec in self.embedding_specs
        ]

    def _apply_split(
        self,
        dev_size: int,
        host_size: int,
        uvm_size: int,
        placements: List[int],
        offsets: List[int],
        enforce_hbm: bool,
    ) -> None:
        assert not self.weight_initialized, "Weights have already been initialized."
        self.weight_initialized = True
        self.weights_physical_placements = placements
        self.weights_physical_offsets = offsets

        self.host_size = host_size
        self.dev_size = dev_size
        self.uvm_size = uvm_size

        self.initialize_logical_weights_placements_and_offsets()

        if dev_size > 0:
            self.weights_dev = torch.zeros(
                dev_size,
                device=self.current_device,
                dtype=torch.uint8,
            )

        if host_size > 0:
            self.weights_host = torch.zeros(
                host_size, device=self.current_device, dtype=torch.uint8
            )

        if uvm_size > 0:
            assert not self.use_cpu
            if enforce_hbm:
                if not torch.jit.is_scripting():
                    logging.info("Enforce hbm for the cache location")
                self.weights_uvm = torch.zeros(
                    uvm_size,
                    device=self.current_device,
                    dtype=torch.uint8,
                )
            else:
                self.weights_uvm = torch.zeros(
                    uvm_size,
                    out=torch.ops.fbgemm.new_unified_tensor(
                        torch.zeros(1, device=self.D_offsets.device, dtype=torch.uint8),
                        [uvm_size],
                        self.uvm_host_mapped,
                    ),
                )

    def _apply_cache_state(
        self,
        cache_state: CacheState,
        cache_algorithm: CacheAlgorithm,
        cache_load_factor: float,
        cache_sets: int,
        cache_reserved_memory: float,
    ) -> None:
        assert self.cache_assoc in [
            1,
            32,
            64,
        ], "Only 1-way or 32-way(64-way for AMD) implmeneted for now"

        self.cache_algorithm = cache_algorithm
        self.timestep_counter = torch.classes.fbgemm.AtomicCounter()
        self.timestep_prefetch_size = torch.classes.fbgemm.AtomicCounter()

        self.max_prefetch_depth = MAX_PREFETCH_DEPTH

        if self.current_device.type == "meta":
            # To reslove "Cannot copy out of meta tensor; no data!" error
            lxu_cache_locations_empty = torch.empty(0, dtype=torch.int32).fill_(-1)
        else:
            lxu_cache_locations_empty = torch.empty(
                0, device=self.current_device, dtype=torch.int32
            ).fill_(-1)
        self.lxu_cache_locations_list = torch.classes.fbgemm.TensorQueue(
            lxu_cache_locations_empty
        )

        # NOTE: no cache for CPU mode!
        if cache_state.total_cache_hash_size == 0 or self.use_cpu:
            self.register_buffer(
                "lxu_cache_weights",
                torch.zeros(0, 0, device=self.current_device, dtype=torch.uint8),
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
                "lxu_cache_miss_timestamp",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "cache_miss_counter",
                torch.tensor([0, 0, 0, 0], dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "uvm_cache_stats",
                torch.zeros(
                    size=(self.uvm_cache_stats_size,),
                    device=self.current_device,
                    dtype=torch.int64,
                ),
                persistent=False,
            )
            self.register_buffer(
                "local_uvm_cache_stats",
                torch.zeros(
                    size=(self.uvm_cache_stats_size,),
                    device=self.current_device,
                    dtype=torch.int32,
                ),
                persistent=False,
            )
            return

        assert cache_load_factor > 0
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
                + self.cache_assoc
                - 1
            ) // self.cache_assoc
            # Note that element_size has been included in max_D_cache (in Bytes)
            cache_size = cache_sets * self.cache_assoc * self.max_D_cache
            if cache_size > free_memory:
                cache_sets = (
                    int(1.0 * free_memory / self.max_D_cache) + self.cache_assoc - 1
                ) // self.cache_assoc
            cache_sets = 1 if cache_sets == 0 else cache_sets
        cache_load_factor = (
            1.0 * cache_sets * self.cache_assoc / int(cache_state.total_cache_hash_size)
        )
        assert cache_sets > 0
        if cache_algorithm == CacheAlgorithm.LFU:
            assert cache_sets < 2**24 - 1
        cache_size = cache_sets * self.cache_assoc * self.max_D_cache
        logging.info(
            f"Using on-device cache with admission algorithm "
            f"{cache_algorithm}, {cache_sets} sets, "
            f"cache_load_factor: {cache_load_factor : .3f}, "
            f"{cache_size / 1024.0 / 1024.0 / 1024.0 : .2f}GB"
        )

        self.total_cache_hash_size = cache_state.total_cache_hash_size
        self.register_buffer(
            "cache_hash_size_cumsum",
            torch.tensor(
                cache_state.cache_hash_size_cumsum,
                device=self.current_device,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "cache_index_table_map",
            torch.tensor(
                cache_state.cache_index_table_map,
                device=self.current_device,
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "lxu_cache_state",
            torch.zeros(
                cache_sets,
                self.cache_assoc,
                device=self.current_device,
                dtype=torch.int64,
            ).fill_(-1),
        )
        self.register_buffer(
            "lxu_cache_weights",
            torch.zeros(
                cache_sets * self.cache_assoc,
                self.max_D_cache,
                device=self.current_device,
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "lxu_state",
            torch.zeros(
                size=(self.total_cache_hash_size + 1,)
                if cache_algorithm == CacheAlgorithm.LFU
                else (cache_sets, self.cache_assoc),
                device=self.current_device,
                dtype=torch.int64,
            ),
        )
        if self.cache_assoc == 1:
            self.register_buffer(
                "lxu_cache_miss_timestamp",
                torch.zeros(
                    cache_sets,
                    self.cache_assoc,
                    device=self.current_device,
                    dtype=torch.int64,
                ),
            )
        else:
            # make TorchScript work
            self.register_buffer(
                "lxu_cache_miss_timestamp",
                torch.zeros(1, device=self.current_device, dtype=torch.int64),
                persistent=False,
            )
        self.register_buffer(
            "cache_miss_counter",
            torch.tensor([0, 0, 0, 0], device=self.current_device, dtype=torch.int64),
        )
        self.register_buffer(
            "uvm_cache_stats",
            torch.zeros(
                size=(self.uvm_cache_stats_size,),
                device=self.current_device,
                dtype=torch.int64,
            ),
            persistent=False,
        )
        self.register_buffer(
            "local_uvm_cache_stats",
            torch.zeros(
                size=(self.uvm_cache_stats_size,),
                device=self.current_device,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        if cache_algorithm not in (CacheAlgorithm.LFU, CacheAlgorithm.LRU):
            raise ValueError(
                f"cache_algorithm must be {CacheAlgorithm.LRU} "
                f"or {CacheAlgorithm.LFU}"
            )

        if self.gather_uvm_cache_stats:
            self.reset_uvm_cache_stats()

    def reset_cache_states(self) -> None:
        if not self.lxu_cache_weights.numel():
            return
        self.lxu_cache_state.fill_(-1)
        self.lxu_state.fill_(0)
        self.timestep_counter.reset()

    @torch.jit.export
    def split_embedding_weights_with_scale_bias(
        self, split_scale_bias_mode: int = 1
    ) -> List[Tuple[Tensor, Optional[Tensor], Optional[Tensor]]]:
        """
        Returns a list of weights, split by table
        split_scale_bias_mode:
            0: return one row;
            1: return weights + scale_bias;
            2: return weights, scale, bias.
        """
        assert self.weight_initialized
        splits: List[Tuple[Tensor, Optional[Tensor], Optional[Tensor]]] = []
        for t, (_, rows, dim, weight_ty, _) in enumerate(self.embedding_specs):
            placement = self.weights_physical_placements[t]
            if placement == EmbeddingLocation.DEVICE.value:
                weights = self.weights_dev
            elif placement == EmbeddingLocation.HOST.value:
                weights = self.weights_host
            else:
                weights = self.weights_uvm
            offset = self.weights_physical_offsets[t]
            weights_shifts = weights.detach()[
                offset : offset
                + rows
                * rounded_row_size_in_bytes(
                    dim, weight_ty, self.row_alignment, self.scale_bias_size_in_bytes
                )
            ].view(
                rows,
                rounded_row_size_in_bytes(
                    dim, weight_ty, self.row_alignment, self.scale_bias_size_in_bytes
                ),
            )

            if split_scale_bias_mode == 1 or split_scale_bias_mode == 2:
                # remove the padding at the end of each row.
                weights_shifts = weights_shifts[
                    :,
                    : unpadded_row_size_in_bytes(
                        dim, weight_ty, self.scale_bias_size_in_bytes
                    ),
                ]
                if (
                    weight_ty.value == SparseType.INT8.value
                    or weight_ty.value == SparseType.INT4.value
                    or weight_ty.value == SparseType.INT2.value
                ):
                    if split_scale_bias_mode == 1:
                        if self.reverse_qparam:
                            splits.append(
                                (
                                    weights_shifts[
                                        :, 0 : (0 - self.scale_bias_size_in_bytes)
                                    ],
                                    weights_shifts[
                                        :, (0 - self.scale_bias_size_in_bytes) :
                                    ],
                                    None,
                                )
                            )
                        else:
                            splits.append(
                                (
                                    weights_shifts[:, self.scale_bias_size_in_bytes :],
                                    weights_shifts[:, : self.scale_bias_size_in_bytes],
                                    None,
                                )
                            )
                    elif split_scale_bias_mode == 2:
                        if self.reverse_qparam:
                            # weights_shifts: [0:-4] is real weights; [-4:-2] is scale; [-2:] is bias
                            splits.append(
                                (
                                    weights_shifts[
                                        :, 0 : (0 - self.scale_bias_size_in_bytes)
                                    ],
                                    weights_shifts[
                                        :,
                                        (0 - self.scale_bias_size_in_bytes) : (
                                            0 - self.scale_bias_size_in_bytes // 2
                                        ),
                                    ].view(torch.float16),
                                    weights_shifts[
                                        :, (0 - self.scale_bias_size_in_bytes // 2) :
                                    ].view(torch.float16),
                                )
                            )
                        else:
                            # weights_shifts: [0:2] is scale; [2:4] is bias; [4:] is real weights
                            splits.append(
                                (
                                    weights_shifts[:, self.scale_bias_size_in_bytes :],
                                    weights_shifts[
                                        :, : self.scale_bias_size_in_bytes // 2
                                    ].view(torch.float16),
                                    weights_shifts[
                                        :,
                                        self.scale_bias_size_in_bytes
                                        // 2 : self.scale_bias_size_in_bytes,
                                    ].view(torch.float16),
                                )
                            )
                    else:
                        raise ValueError("split_scale_bias_mode is not supported")

                elif (
                    weight_ty.value == SparseType.FP8.value
                    or weight_ty.value == SparseType.FP16.value
                    or weight_ty.value == SparseType.FP32.value
                ):
                    splits.append(
                        (
                            weights_shifts,
                            None,
                            None,
                        )
                    )
                else:
                    raise ValueError("weight_ty is not supported")

            else:  # split_scale_bias_mode == 0:
                splits.append((weights_shifts, None, None))

        return splits

    @torch.jit.export
    def split_embedding_weights(
        self,
        split_scale_shifts: bool = True
        # When true, return list of two tensors, the first with weights and
        # the second with scale_bias.
        # This should've been named as split_scale_bias.
        # Keep as is for backward compatibility.
    ) -> List[Tuple[Tensor, Optional[Tensor]]]:
        """
        Returns a list of weights, split by table
        """
        splits: List[
            Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        ] = self.split_embedding_weights_with_scale_bias(
            split_scale_bias_mode=(1 if split_scale_shifts else 0)
        )
        return [
            (split_weight_scale_bias[0], split_weight_scale_bias[1])
            for split_weight_scale_bias in splits
        ]

    @torch.jit.export
    def initialize_weights(self) -> None:
        if not self.weight_initialized:
            self._apply_split(
                self.dev_size,
                self.host_size,
                self.uvm_size,
                self.weights_physical_placements,
                self.weights_physical_offsets,
                self.enforce_hbm,
            )
            self.weight_initialized = True

    def fill_random_weights(self) -> None:
        """
        Fill the buffer with random weights, table by table
        FIXME: make it in-place fill.
        """
        self.initialize_weights()
        weights = self.split_embedding_weights()
        for dest_weight in weights:
            dest_weight[0].copy_(
                random_quant_scaled_tensor(
                    shape=dest_weight[0].shape, device=self.current_device
                )
            )

    def assign_embedding_weights(
        self, q_weight_list: List[Tuple[Tensor, Optional[Tensor]]]
    ) -> None:
        """
        Assigns self.split_embedding_weights() with values from the input list of weights and scale_shifts.
        """
        weights = self.split_embedding_weights()
        assert len(q_weight_list) == len(weights)

        for dest_weight, input_weight in zip(weights, q_weight_list):
            dest_weight[0].copy_(input_weight[0])
            if input_weight[1] is not None:
                assert dest_weight[1] is not None
                dest_weight[1].copy_(input_weight[1])
            else:
                assert dest_weight[1] is None

    @torch.jit.export
    def set_index_remappings_array(
        self,
        index_remapping: List[Tensor],
    ) -> None:
        rows: List[int] = [e[1] for e in self.embedding_specs]
        index_remappings_array_offsets = [0]
        original_feature_rows = torch.jit.annotate(List[int], [])
        last_offset = 0
        for t, mapping in enumerate(index_remapping):
            if mapping is not None:
                current_original_row = mapping.numel()
                last_offset += current_original_row
                original_feature_rows.append(current_original_row)
            else:
                original_feature_rows.append(rows[t])
            index_remappings_array_offsets.append(last_offset)

        self.index_remappings_array_offsets = torch.tensor(
            index_remappings_array_offsets,
            device=self.current_device,
            dtype=torch.int64,
        )
        if len(original_feature_rows) == 0:
            original_feature_rows = rows
        self.original_rows_per_table = torch.tensor(
            [original_feature_rows[t] for t in self.feature_table_map],
            device=self.current_device,
            dtype=torch.int64,
        )
        if self.index_remappings_array_offsets[-1] == 0:
            self.index_remappings_array = torch.empty(
                0, dtype=torch.int32, device=self.current_device
            )
        else:
            index_remappings_filter_nones = []
            for mapping in index_remapping:
                if mapping is not None:
                    index_remappings_filter_nones.append(mapping)
            self.index_remappings_array = torch.cat(index_remappings_filter_nones).to(
                self.current_device
            )

    def set_index_remappings(
        self,
        index_remapping: List[Tensor],
        pruning_hash_load_factor: float = 0.5,
        use_array_for_index_remapping: bool = True,
    ) -> None:
        rows: List[int] = [e[1] for e in self.embedding_specs]
        T = len(self.embedding_specs)
        # Hash mapping pruning
        if not use_array_for_index_remapping:
            capacities = [
                round_up(int(row * 1.0 / pruning_hash_load_factor), 32)
                if index_remap is not None
                else 0
                for (index_remap, row) in zip(index_remapping, rows)
            ]
            hash_table = torch.empty(
                (sum(capacities), 2),
                dtype=torch.int32,
            )
            hash_table[:, :] = -1
            hash_table_offsets = torch.tensor([0] + list(accumulate(capacities))).long()

            merged_index_remappings = [
                mapping if mapping is not None else Tensor(list(range(row)))
                for (mapping, row) in zip(index_remapping, rows)
            ]
            original_feature_rows = [
                mapping.numel() for mapping in merged_index_remappings
            ]
            if len(original_feature_rows) == 0:
                original_feature_rows = rows
            self.original_rows_per_table = torch.tensor(
                [original_feature_rows[t] for t in self.feature_table_map],
                device=self.current_device,
                dtype=torch.int64,
            )
            dense_indices = torch.cat(merged_index_remappings, dim=0).int()
            indices = torch.cat(
                [torch.arange(row) for row in original_feature_rows], dim=0
            ).int()
            offsets = torch.tensor([0] + list(accumulate(original_feature_rows))).int()

            if self.use_cpu:
                self.index_remapping_hash_table_cpu = (
                    torch.classes.fbgemm.PrunedMapCPU()
                )
                self.index_remapping_hash_table_cpu.insert(
                    indices, dense_indices, offsets, T
                )
            else:
                # pruned_hashmap_insert only has cpu implementation: Move dense_indices to CPU
                torch.ops.fbgemm.pruned_hashmap_insert(
                    indices,
                    dense_indices.cpu(),
                    offsets,
                    hash_table,
                    hash_table_offsets,
                )
                self.index_remapping_hash_table = hash_table.to(self.current_device)
                self.index_remapping_hash_table_offsets = hash_table_offsets.to(
                    self.current_device
                )
                self.index_remapping_hash_table_cpu = None
        # Array mapping pruning
        else:
            self.set_index_remappings_array(index_remapping)

    def _embedding_inplace_update_per_table(
        self,
        update_table_idx: int,
        update_row_indices: List[int],
        update_weights: Tensor,
    ) -> None:
        row_size = len(update_row_indices)
        if row_size == 0:
            return
        # pyre-fixme[9]: update_row_indices has type `List[int]`; used as `Tensor`.
        update_row_indices = torch.tensor(
            update_row_indices,
            device=self.current_device,
            dtype=torch.int64,
        )
        table_values = self.split_embedding_weights(split_scale_shifts=False)[
            update_table_idx
        ]
        table_values[0].scatter_(
            dim=0,
            # pyre-fixme[16]: `List` has no attribute `view`.
            index=update_row_indices.view(row_size, 1).expand_as(update_weights),
            src=update_weights,
        )

    @torch.jit.export
    def embedding_inplace_update(
        self,
        update_table_indices: List[int],
        update_row_indices: List[List[int]],
        update_weights: List[Tensor],
    ) -> None:
        for i in range(len(update_table_indices)):
            self._embedding_inplace_update_per_table(
                update_table_indices[i],
                update_row_indices[i],
                update_weights[i],
            )

    def embedding_inplace_update_internal(
        self,
        update_table_indices: List[int],
        update_row_indices: List[int],
        update_weights: Tensor,
    ) -> None:
        assert len(update_table_indices) == len(update_row_indices)
        update_offsets = []
        update_offset = 0
        for table_idx in update_table_indices:
            D_bytes = rounded_row_size_in_bytes(
                self.embedding_specs[table_idx][2],
                self.embedding_specs[table_idx][3],
                self.row_alignment,
                self.scale_bias_size_in_bytes,
            )
            update_offsets.append(update_offset)
            update_offset += D_bytes
        update_offsets.append(update_offset)

        # pyre-fixme[9]: update_table_indices has type `List[int]`; used as `Tensor`.
        update_table_indices = torch.tensor(
            update_table_indices,
            device=self.current_device,
            dtype=torch.int32,
        )
        # pyre-fixme[9]: update_row_indices has type `List[int]`; used as `Tensor`.
        update_row_indices = torch.tensor(
            update_row_indices,
            device=self.current_device,
            dtype=torch.int64,
        )
        update_offsets = torch.tensor(
            update_offsets,
            device=self.current_device,
            dtype=torch.int64,
        )

        # Only support array based pruning for now.
        assert self.index_remapping_hash_table_cpu is None
        assert self.index_remapping_hash_table.numel() == 0
        assert self.index_remappings_array.numel() >= 0

        if self.index_remappings_array.numel() > 0:
            update_row_indices = torch.ops.fbgemm.pruned_array_lookup_from_row_idx(
                update_row_indices,
                update_table_indices,
                self.index_remappings_array,
                self.index_remappings_array_offsets,
            )

        lxu_cache_locations = None
        if self.lxu_cache_weights.numel() > 0:
            linear_cache_indices = (
                torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
                    self.cache_hash_size_cumsum,
                    update_table_indices,
                    update_row_indices,
                )
            )

            if self.cache_assoc in [32, 64]:
                # 64 for AMD
                self.prefetch_32way(linear_cache_indices)
            elif self.cache_assoc == 1:
                self.prefetch_1way(linear_cache_indices)
            else:
                raise ValueError(f"{self.cache_assoc} not in [1, 32, 64]")

            lxu_cache_locations = self.lxu_cache_locations_list.pop()

        torch.ops.fbgemm.emb_inplace_update(
            dev_weights=self.weights_host if self.host_size > 0 else self.weights_dev,
            uvm_weights=self.weights_uvm,
            weights_placements=self.weights_placements,
            weights_offsets=self.weights_offsets,
            weights_tys=self.weights_tys,
            D_offsets=self.D_offsets,
            update_weights=update_weights,
            update_table_indices=update_table_indices,
            update_row_indices=update_row_indices,
            update_offsets=update_offsets,
            row_alignment=self.row_alignment,
            lxu_cache_weights=self.lxu_cache_weights,
            lxu_cache_locations=lxu_cache_locations,
        )
