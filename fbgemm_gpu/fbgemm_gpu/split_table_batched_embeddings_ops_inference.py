#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import logging
import uuid
from itertools import accumulate
from typing import List, Optional, Tuple, Union

import fbgemm_gpu  # noqa: F401
import torch  # usort:skip
from torch import nn, Tensor  # usort:skip

from fbgemm_gpu.config import FeatureGateName
from fbgemm_gpu.split_embedding_configs import sparse_type_to_int, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheState,
    construct_cache_state,
    DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    EmbeddingLocation,
    EmbeddingSpecInfo,
    get_bounds_check_version_for_platform,
    get_new_embedding_location,
    MAX_PREFETCH_DEPTH,
    PoolingMode,
    RecordCacheMetrics,
    round_up,
    SplitState,
    tensor_to_device,
)
from fbgemm_gpu.utils.loader import load_torch_module, load_torch_module_bc

try:
    load_torch_module(
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_inference_gpu",
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cuda_inference",
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_hip_inference",
    )
except Exception:
    pass

try:
    load_torch_module_bc(
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_inference_cpu",
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cpu_inference",
    )
except Exception:
    pass

import fbgemm_gpu  # noqa


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
        elif location == EmbeddingLocation.DEVICE or location == EmbeddingLocation.MTIA:
            placements.append(location)
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


def random_quant_scaled_tensor(
    shape: torch.Size,
    device: torch.device,
    output_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if output_tensor is not None:
        return torch.randint(
            0,
            255,
            size=shape,
            out=output_tensor,
            dtype=torch.uint8,
            device=device,
        )
    else:
        return torch.randint(
            0,
            255,
            size=shape,
            dtype=torch.uint8,
            device=device,
        )


@torch.fx.wrap
def inputs_to_device(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    per_sample_weights: Optional[torch.Tensor],
    bounds_check_warning: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if bounds_check_warning.device.type == "meta":
        return indices, offsets, per_sample_weights

    non_blocking = bounds_check_warning.device.type != "cpu"
    if indices.device != bounds_check_warning.device:
        indices = indices.to(bounds_check_warning.device, non_blocking=non_blocking)
    if offsets.device != bounds_check_warning.device:
        offsets = offsets.to(bounds_check_warning.device, non_blocking=non_blocking)
    if (
        per_sample_weights is not None
        and per_sample_weights.device != bounds_check_warning.device
    ):
        per_sample_weights = per_sample_weights.to(
            bounds_check_warning.device, non_blocking=non_blocking
        )
    return indices, offsets, per_sample_weights


# pyre-fixme[13]: Attribute `cache_miss_counter` is never initialized.
class IntNBitTableBatchedEmbeddingBagsCodegen(nn.Module):
    """
    Table-batched version of nn.EmbeddingBag(sparse=False)
    Inference version, with support for FP32/FP16/FP8/INT8/INT4/INT2 weights

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

        index_remapping (Optional[List[Tensor]] = None): Index remapping for pruning

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

        weight_lists (Optional[List[Tuple[Tensor, Optional[Tensor]]]] = None):
            [T]

        pruning_hash_load_factor (float = 0.5):
            Load factor for pruning hash

        use_array_for_index_remapping (bool = True):
            If True, use array for index remapping. Otherwise, use hash map.

        output_dtype (SparseType = SparseType.FP16): The data type of an output
            tensor.

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

        enforce_hbm (bool = False): If True, place all weights/momentums in HBM
            when using `EmbeddingLocation.MANAGED_CACHING`

        record_cache_metrics (Optional[RecordCacheMetrics] = None): Record
            a number of hits, a number of requests, etc if
            `RecordCacheMetrics.record_cache_miss_counter` is True and record
            the similar metrics table-wise if
            `RecordCacheMetrics.record_tablewise_cache_miss is True`

        gather_uvm_cache_stats (Optional[bool] = False): If True, collect the
            cache statistics when `EmbeddingLocation` is set to
            `MANAGED_CACHING`

        row_alignment (Optional[int] = None): Row alignment

        fp8_exponent_bits (Optional[int] = None): Exponent bits when using FP8

        fp8_exponent_bias (Optional[int] = None): Exponent bias when using FP8

        cache_assoc (int = 32): Number of ways for cache

        scale_bias_size_in_bytes (int = DEFAULT_SCALE_BIAS_SIZE_IN_BYTES): Size
            of scale and bias in bytes

        cacheline_alignment (bool = True): If True, align each table to 128b
            cache line boundary

        uvm_host_mapped (bool = False): If True, allocate every UVM tensor
            using `malloc` + `cudaHostRegister`. Otherwise use
            `cudaMallocManaged`

        reverse_qparam (bool = False): If True, load `qparams` at end of each
            row.  Otherwise, load `qparams` at begnning of each row.

        feature_names_per_table (Optional[List[List[str]]] = None): An optional
            list that specifies feature names per table. `feature_names_per_table[t]`
            indicates the feature names of table `t`.

        indices_dtype (torch.dtype = torch.int32): The expected dtype of the
            indices tensor that will be passed to the `forward()` call.  This
            information will be used to construct the remap_indices array/hash.
            Options are `torch.int32` and `torch.int64`.
    """

    embedding_specs: List[Tuple[str, int, int, SparseType, EmbeddingLocation]]
    record_cache_metrics: RecordCacheMetrics
    # pyre-fixme[13]: Attribute `cache_miss_counter` is never initialized.
    cache_miss_counter: torch.Tensor
    # pyre-fixme[13]: Attribute `uvm_cache_stats` is never initialized.
    uvm_cache_stats: torch.Tensor
    # pyre-fixme[13]: Attribute `local_uvm_cache_stats` is never initialized.
    local_uvm_cache_stats: torch.Tensor
    # pyre-fixme[13]: Attribute `weights_offsets` is never initialized.
    weights_offsets: torch.Tensor
    # pyre-fixme[13]: Attribute `weights_placements` is never initialized.
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
        feature_names_per_table: Optional[List[List[str]]] = None,
        indices_dtype: torch.dtype = torch.int32,  # Used for construction of the remap_indices tensors.  Should match the dtype of the indices passed in the forward() call (INT32 or INT64).
    ) -> None:  # noqa C901  # tuple of (rows, dims,)
        super(IntNBitTableBatchedEmbeddingBagsCodegen, self).__init__()
        self.uuid = str(uuid.uuid4())
        self.log(
            f"Feature Gates: {[(feature.name, feature.is_enabled()) for feature in FeatureGateName]}"
        )

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
        self.feature_names_per_table = feature_names_per_table
        self.indices_dtype = indices_dtype
        # (feature_names, rows, dims, weights_tys, locations) = zip(*embedding_specs)
        # Pyre workaround
        self.feature_names: List[str] = [e[0] for e in embedding_specs]
        self.cache_load_factor: float = cache_load_factor
        self.cache_sets: int = cache_sets
        self.cache_reserved_memory: float = cache_reserved_memory
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
                [
                    dim
                    for dim, weight_ty in zip(dims, weights_tys)
                    if weight_ty == ty or weight_ty.value == ty.value
                ],
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

        self.weights_uvm: torch.Tensor = torch.empty(
            0, device=self.current_device, dtype=torch.uint8
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
        # All buffers are int64 in order to support both int32 and int64 indices.
        self.register_buffer(
            "index_remappings_array_offsets",
            torch.empty(0, device=self.current_device, dtype=torch.int64),
        )
        self.register_buffer(
            "index_remappings_array",
            torch.empty(0, device=self.current_device, dtype=self.indices_dtype),
        )
        self.register_buffer(
            "index_remapping_hash_table_offsets",
            torch.empty(0, device=self.current_device, dtype=torch.int64),
        )
        self.register_buffer(
            "index_remapping_hash_table",
            torch.empty(0, device=self.current_device, dtype=self.indices_dtype),
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

        self.bounds_check_version: int = get_bounds_check_version_for_platform()

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

    @torch.jit.export
    def get_feature_num_per_table(self) -> List[int]:
        if self.feature_names_per_table is None:
            return []
        return [len(feature_names) for feature_names in self.feature_names_per_table]

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
        self.log(
            f"\n"
            f"Miss counter value [0] - # of miss occured iters : {self.cache_miss_counter[0]}, \n"
            f"Miss counter value [1] - # of unique misses : {self.cache_miss_counter[1]}, \n"
            f"Miss counter value [2] - # of unique requested indices : {self.cache_miss_counter[2]}, \n"
            f"Miss counter value [3] - # of total requested indices : {self.cache_miss_counter[3]}, "
        )
        self.log(
            f"unique_miss_rate using counter : {self.cache_miss_counter[1] / self.cache_miss_counter[2]}, \n"
        )
        self.log(
            f"total_miss_rate using counter : {self.cache_miss_counter[1] / self.cache_miss_counter[3]}, \n"
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
        self.log(
            f"N_called: {uvm_cache_stats[0]}\n"
            f"N_requested_indices: {uvm_cache_stats[1]}\n"
            f"N_unique_indices: {uvm_cache_stats[2]}\n"
            f"N_unique_misses: {uvm_cache_stats[3]}\n"
            f"N_conflict_unique_misses: {uvm_cache_stats[4]}\n"
            f"N_conflict_misses: {uvm_cache_stats[5]}\n"
        )
        if uvm_cache_stats[1]:
            self.log(
                f"unique indices / requested indices: {uvm_cache_stats[2] / uvm_cache_stats[1]}\n"
                f"unique misses / requested indices: {uvm_cache_stats[3] / uvm_cache_stats[1]}\n"
            )

    @torch.jit.export
    def prefetch(self, indices: Tensor, offsets: Tensor) -> None:
        self.timestep_counter.increment()
        self.timestep_prefetch_size.increment()
        # pyre-fixme[29]: `Union[(self: TensorBase) -> int, Module, Tensor]` is not
        #  a function.
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

    def _forward_impl(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        assert (
            self.weight_initialized
        ), "weight needs to be initialized before forward function"

        indices, offsets, per_sample_weights = inputs_to_device(
            indices, offsets, per_sample_weights, self.bounds_check_warning
        )

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
                    bounds_check_version=self.bounds_check_version,
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
        # pyre-fixme[29]: `Union[(self: TensorBase) -> int, Module, Tensor]` is not
        #  a function.
        if self.lxu_cache_weights.numel() > 0:
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
                bounds_check_version=self.bounds_check_version,
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

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        return self._forward_impl(
            indices=indices, offsets=offsets, per_sample_weights=per_sample_weights
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
        # Overwrite location in embedding_specs with new location
        # Use map since can't script enum call (ie. EmbeddingLocation(value))
        INT_TO_EMBEDDING_LOCATION = {
            EmbeddingLocation.DEVICE.value: EmbeddingLocation.DEVICE,
            EmbeddingLocation.MANAGED.value: EmbeddingLocation.MANAGED,
            EmbeddingLocation.MANAGED_CACHING.value: EmbeddingLocation.MANAGED_CACHING,
            EmbeddingLocation.HOST.value: EmbeddingLocation.HOST,
            EmbeddingLocation.MTIA.value: EmbeddingLocation.MTIA,
        }
        # Reset device/location denoted in embedding specs
        target_location = INT_TO_EMBEDDING_LOCATION[location]
        if target_location == EmbeddingLocation.MTIA:
            self.scale_bias_size_in_bytes = 8
        self.reset_embedding_spec_location(device, target_location)
        # Initialize all physical/logical weights placements and offsets without initializing large dev weights tensor
        self.initialize_physical_weights_placements_and_offsets(
            cacheline_alignment=target_location != EmbeddingLocation.MTIA
        )
        self.initialize_logical_weights_placements_and_offsets()

    def reset_embedding_spec_location(
        self, device: torch.device, target_location: EmbeddingLocation
    ) -> None:
        self.current_device = device
        self.row_alignment = (
            1
            if target_location == EmbeddingLocation.HOST
            or target_location == EmbeddingLocation.MTIA
            else 16
        )
        self.embedding_specs = [
            (spec[0], spec[1], spec[2], spec[3], target_location)
            for spec in self.embedding_specs
        ]

    @torch.jit.export
    def recompute_module_buffers(self) -> None:
        """
        Compute module buffers that're on meta device and are not materialized
        in reset_weights_placements_and_offsets().  Currently those buffers are
        `weights_tys`, `rows_per_table`, `D_offsets` and `bounds_check_warning`.
        Pruning related or uvm related buffers are not computed right now.
        """
        if (
            self.weights_tys.device == self.current_device
            or self.current_device.type == "meta"
        ):
            return

        weights_tys_int = [sparse_type_to_int(e[3]) for e in self.embedding_specs]
        self.weights_tys = torch.tensor(
            [weights_tys_int[t] for t in self.feature_table_map],
            device=self.current_device,
            dtype=torch.uint8,
        )
        rows = [e[1] for e in self.embedding_specs]
        self.rows_per_table = torch.tensor(
            [rows[t] for t in self.feature_table_map],
            device=self.current_device,
            dtype=torch.int64,
        )
        dims = [e[2] for e in self.embedding_specs]
        D_offsets_list = [0]
        for t in self.feature_table_map:
            D_offsets_list.append(dims[t] + D_offsets_list[-1])
        self.D_offsets = torch.tensor(
            D_offsets_list, device=self.current_device, dtype=torch.int32
        )
        self.bounds_check_warning = torch.tensor(
            [0], device=self.current_device, dtype=torch.int64
        )

        # For pruning related or uvm related buffers, we just set them as empty tensors.
        self.index_remapping_hash_table = torch.empty_like(
            self.index_remapping_hash_table, device=self.current_device
        )
        self.index_remapping_hash_table_offsets = torch.empty_like(
            self.index_remapping_hash_table_offsets, device=self.current_device
        )
        self.index_remappings_array = torch.empty_like(
            self.index_remappings_array, device=self.current_device
        )
        self.index_remappings_array_offsets = torch.empty_like(
            self.index_remappings_array_offsets, device=self.current_device
        )
        # pyre-fixme[16]: `IntNBitTableBatchedEmbeddingBagsCodegen` has no attribute
        #  `lxu_cache_weights`.
        self.lxu_cache_weights = torch.empty_like(
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            self.lxu_cache_weights,
            device=self.current_device,
        )
        self.original_rows_per_table = torch.empty_like(
            self.original_rows_per_table, device=self.current_device
        )
        self.table_wise_cache_miss = torch.empty_like(
            self.table_wise_cache_miss, device=self.current_device
        )
        self.weights_uvm = torch.empty_like(
            self.weights_uvm, device=self.current_device
        )

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
                    self.log("Enforce hbm for the cache location")
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
        # pyre-ignore[16]
        self.timestep_counter = torch.classes.fbgemm.AtomicCounter()
        # pyre-ignore[16]
        self.timestep_prefetch_size = torch.classes.fbgemm.AtomicCounter()

        self.max_prefetch_depth = MAX_PREFETCH_DEPTH

        if self.current_device.type == "meta":
            # To reslove "Cannot copy out of meta tensor; no data!" error
            lxu_cache_locations_empty = torch.empty(0, dtype=torch.int32).fill_(-1)
        else:
            lxu_cache_locations_empty = torch.empty(
                0, device=self.current_device, dtype=torch.int32
            ).fill_(-1)
        # pyre-ignore[16]
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
            self.total_cache_hash_size = 0
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
                torch.tensor(
                    [0, 0, 0, 0], dtype=torch.int64, device=self.current_device
                ),
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
        self.log(
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
                size=(
                    (self.total_cache_hash_size + 1,)
                    if cache_algorithm == CacheAlgorithm.LFU
                    else (cache_sets, self.cache_assoc)
                ),
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
        # pyre-fixme[29]: `Union[(self: TensorBase) -> int, Module, Tensor]` is not
        #  a function.
        if not self.lxu_cache_weights.numel():
            return
        self.lxu_cache_state.fill_(-1)
        self.lxu_state.fill_(0)
        self.timestep_counter.reset()

    def move_to_device_with_cache(
        self, device: torch.device, cache_load_factor: float
    ) -> None:
        """
        Moves the TBE to the specified device, and updates the cache state accordingly.
        """
        if (
            self.current_device == device
            and self.cache_load_factor == cache_load_factor
        ):
            return

        location = get_new_embedding_location(device, cache_load_factor)
        if device.type != "cpu":
            self.use_cpu = False

        weights = self.split_embedding_weights()
        is_meta = self.current_device.type == "meta"
        index_remapping_array: torch.Tensor
        index_remappings_array_offsets: torch.Tensor
        original_rows_per_table: torch.Tensor
        if not is_meta:
            # Record weights and pruning tensors for setting
            # weights and pruning tensors for TBE on new device
            if device.type == "cpu":
                for i, weight in enumerate(weights):
                    weights[i] = (
                        weight[0].to(device),
                        weight[1].to(device) if weight[1] is not None else None,
                    )
            (
                index_remapping_array,
                index_remappings_array_offsets,
                original_rows_per_table,
            ) = (
                self.index_remappings_array.to(device),
                self.index_remappings_array_offsets.to(device),
                self.original_rows_per_table.to(device),
            )

        self.reset_weights_placements_and_offsets(device, location.value)
        self.recompute_module_buffers()
        self.weight_initialized = False
        self.initialize_weights()

        # Ensure all weights are on the same device
        if device.type != "cpu":
            self.weights_host = torch.zeros(0, device=device, dtype=torch.uint8)

        if location != EmbeddingLocation.DEVICE:
            self.weights_dev = torch.zeros(0, device=device, dtype=torch.uint8)

        for name, buf in self.named_buffers():
            if buf.is_meta:
                self.register_buffer(name, tensor_to_device(buf, device))

        self.current_device = device

        if not is_meta:
            self.assign_embedding_weights(weights)
            self.index_remappings_array = index_remapping_array
            self.index_remappings_array_offsets = index_remappings_array_offsets
            self.original_rows_per_table = original_rows_per_table

        if cache_load_factor is not None:
            self.update_cache_load_factor(cache_load_factor)

    def update_cache_load_factor(self, cache_load_factor: float = 0.2) -> None:
        """
        Updates cache_load_factor and embedding location for weights after TBE has already been initialized
        Assumes that the location of the weights is already set correctly
        """
        rows = [
            embedding_spec[EmbeddingSpecInfo.rows]
            for embedding_spec in self.embedding_specs
        ]
        locations = [
            embedding_spec[EmbeddingSpecInfo.embedding_location]
            for embedding_spec in self.embedding_specs
        ]
        # pyre-ignore[6]
        cache_state = construct_cache_state(rows, locations, self.feature_table_map)

        cached_dims = [
            rounded_row_size_in_bytes(
                embedding_spec[EmbeddingSpecInfo.dims],  # pyre-ignore[6]
                embedding_spec[EmbeddingSpecInfo.sparse_type],  # pyre-ignore[6]
                16,
                self.scale_bias_size_in_bytes,
            )
            for embedding_spec in self.embedding_specs
            if embedding_spec[EmbeddingSpecInfo.embedding_location]
            == EmbeddingLocation.MANAGED_CACHING
        ]

        self.max_D_cache: int = max(cached_dims) if len(cached_dims) > 0 else 0

        self._apply_cache_state(
            cache_state,
            self.cache_algorithm,
            cache_load_factor,
            self.cache_sets,
            self.cache_reserved_memory,
        )

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
            if (
                placement == EmbeddingLocation.DEVICE.value
                or placement == EmbeddingLocation.MTIA.value
            ):
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
        split_scale_shifts: bool = True,
        # When true, return list of two tensors, the first with weights and
        # the second with scale_bias.
        # This should've been named as split_scale_bias.
        # Keep as is for backward compatibility.
    ) -> List[Tuple[Tensor, Optional[Tensor]]]:
        """
        Returns a list of weights, split by table
        """
        # fmt: off
        splits: List[Tuple[Tensor, Optional[Tensor], Optional[Tensor]]] = (
            self.split_embedding_weights_with_scale_bias(
                split_scale_bias_mode=(1 if split_scale_shifts else 0)
            )
        )
        # fmt: on
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
        """
        self.initialize_weights()
        weights = self.split_embedding_weights()
        for dest_weight in weights:
            random_quant_scaled_tensor(
                shape=dest_weight[0].shape,
                device=self.current_device,
                output_tensor=dest_weight[0],
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

        index_remappings_filter_nones = []
        for mapping in index_remapping:
            if mapping is not None:
                index_remappings_filter_nones.append(mapping)
        if len(index_remappings_filter_nones) == 0:
            self.index_remappings_array = torch.empty(
                0, dtype=self.indices_dtype, device=self.current_device
            )
        else:
            self.index_remappings_array = torch.cat(index_remappings_filter_nones).to(
                dtype=self.indices_dtype, device=self.current_device
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
                (
                    round_up(int(row * 1.0 / pruning_hash_load_factor), 32)
                    if index_remap is not None
                    else 0
                )
                for (index_remap, row) in zip(index_remapping, rows)
            ]
            hash_table = torch.empty(
                (sum(capacities), 2),
                dtype=self.indices_dtype,
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
                    # pyre-ignore[16]
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
                self.index_remapping_hash_table = hash_table.to(
                    dtype=self.indices_dtype, device=self.current_device
                )
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
        # pyre-fixme[29]: `Union[(self: TensorBase) -> int, Module, Tensor]` is not
        #  a function.
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
