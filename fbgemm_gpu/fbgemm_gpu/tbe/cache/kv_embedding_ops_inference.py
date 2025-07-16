#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]


from typing import List, Optional, Tuple, Union

import torch  # usort:skip
from torch import Tensor  # usort:skip
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    EmbeddingLocation,
    PoolingMode,
    RecordCacheMetrics,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    inputs_to_device,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    random_quant_scaled_tensor,
    rounded_row_size_in_bytes,
)
from fbgemm_gpu.utils.loader import load_torch_module

try:
    load_torch_module(
        "//deeplearning/fbgemm/fbgemm_gpu:dram_kv_embedding_inference",
    )
except Exception:
    pass


class KVEmbeddingInference(IntNBitTableBatchedEmbeddingBagsCodegen):
    """
    KV Table-batched version of nn.EmbeddingBag(sparse=False)
    Inference version, with support for FP32/FP16/FP8/INT8/INT4/INT2 weights
    """

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
        super(KVEmbeddingInference, self).__init__(
            embedding_specs=embedding_specs,
            feature_table_map=feature_table_map,
            index_remapping=index_remapping,
            pooling_mode=pooling_mode,
            device=device,
            bounds_check_mode=bounds_check_mode,
            weight_lists=weight_lists,
            pruning_hash_load_factor=pruning_hash_load_factor,
            use_array_for_index_remapping=use_array_for_index_remapping,
            output_dtype=output_dtype,
            cache_algorithm=cache_algorithm,
            cache_load_factor=cache_load_factor,
            cache_sets=cache_sets,
            cache_reserved_memory=cache_reserved_memory,
            enforce_hbm=enforce_hbm,
            record_cache_metrics=record_cache_metrics,
            gather_uvm_cache_stats=gather_uvm_cache_stats,
            row_alignment=row_alignment,
            fp8_exponent_bits=fp8_exponent_bits,
            fp8_exponent_bias=fp8_exponent_bias,
            cache_assoc=cache_assoc,
            scale_bias_size_in_bytes=scale_bias_size_in_bytes,
            cacheline_alignment=cacheline_alignment,
            uvm_host_mapped=uvm_host_mapped,
            reverse_qparam=reverse_qparam,
            feature_names_per_table=feature_names_per_table,
            indices_dtype=indices_dtype,
        )
        self.register_buffer(
            "weights_ids",
            torch.tensor(0, device=self.current_device, dtype=torch.int64),
        )

        num_shards = 32
        uniform_init_lower: float = -0.01
        uniform_init_upper: float = 0.01
        # pyre-fixme[4]: Attribute must be annotated.
        self.kv_embedding_cache = torch.classes.fbgemm.DramKVEmbeddingInferenceWrapper(
            num_shards, uniform_init_lower, uniform_init_upper
        )

        self.specs: List[Tuple[int, int, int]] = [
            (rows, dims, sparse_type.as_int())
            for (_, rows, dims, sparse_type, _) in self.embedding_specs
        ]
        # table shard offset if inference sharding is enabled, otherwise, should be all zeros
        self.table_sharding_offset: List[int] = [0] * len(self.embedding_specs)
        self.kv_embedding_cache_initialized = False
        self.hash_size_cumsum: torch.Tensor = torch.zeros(
            0,
            device=self.current_device,
            dtype=torch.int64,
        )
        self.feature_hash_size_cumsum: torch.Tensor = torch.zeros(
            0,
            device=self.current_device,
            dtype=torch.int64,
        )

    def construct_hash_size_cumsum(self) -> List[int]:
        hash_size_cumsum = [0]
        for spec in self.embedding_specs:
            rows = spec[1]
            hash_size_cumsum.append(hash_size_cumsum[-1] + rows)
        return hash_size_cumsum

    def calculate_indices_and_weights_offsets(
        self, indices: Tensor, offsets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.pooling_mode is not PoolingMode.NONE:
            T = self.weights_offsets.numel()
        else:
            T = self.D_offsets.numel() - 1
        B = int((offsets.size(0) - 1) / T)

        total_bytes_added = 0
        new_indices = torch.tensor(
            [0] * indices.size(0), device=self.current_device, dtype=indices.dtype
        )
        new_weights_offsets = torch.tensor(
            [0] * T, device=self.current_device, dtype=self.weights_offsets.dtype
        )
        for t in range(T):
            new_weights_offsets[t] = total_bytes_added
            start, end = int(offsets[t * B]), int(offsets[(t + 1) * B])
            index_size = end - start
            new_indices[start:end] = torch.arange(index_size)
            table_id = self.feature_table_map[t]
            total_bytes_added += index_size * rounded_row_size_in_bytes(
                self.embedding_specs[table_id][2],  # dim
                self.embedding_specs[table_id][3],  # weight_ty
                self.row_alignment,
                self.scale_bias_size_in_bytes,
            )
        return new_indices, new_weights_offsets

    def linearize_cache_indices(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linearize cache indices for KV cache.
        """
        linearized_indices = torch.zeros(
            indices.numel(),
            device=indices.device,
            dtype=torch.int64,
        )

        T = self.feature_hash_size_cumsum.numel() - 1
        B = int((offsets.size(0) - 1) / T)

        for t in range(T):
            start, end = int(offsets[t * B]), int(offsets[(t + 1) * B])
            linearized_indices[start:end] = (
                indices[start:end] + self.feature_hash_size_cumsum[t]
            )

        return linearized_indices

    def forward(
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

        lxu_cache_locations = self.lxu_cache_locations_list.pop()

        weights_offsets = self.weights_offsets
        weights = self.weights_host if self.host_size > 0 else self.weights_dev

        if self.kv_embedding_cache_initialized:
            indices = self.linearize_cache_indices(
                indices,
                offsets,
            )

            weights = self.kv_embedding_cache.get_embeddings(indices)

            indices, weights_offsets = self.calculate_indices_and_weights_offsets(
                indices, offsets
            )

        return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
            dev_weights=weights,
            uvm_weights=self.weights_uvm,
            weights_placements=self.weights_placements,
            weights_offsets=weights_offsets,
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

    def fill_random_weights(self) -> None:
        """
        Fill the buffer with random weights, table by table
        """
        self.initialize_kv_embedding_cache()
        for i, (_, num_embeddings, embedding_dim, weight_ty, _) in enumerate(
            self.embedding_specs
        ):
            embedding_dim = rounded_row_size_in_bytes(
                embedding_dim, weight_ty, self.row_alignment
            )
            indices = torch.range(0, num_embeddings - 1, dtype=torch.int64)
            weights = random_quant_scaled_tensor(
                shape=torch.Size([num_embeddings, embedding_dim]),
                device=self.current_device,
            )
            self.embedding_inplace_update_per_table(
                i,
                indices,
                weights,
            )
        self.weight_initialized = True

    @torch.jit.export
    def init_tbe_config(self, table_sharding_offset: List[int]) -> None:
        """
        Initialize the dynamic TBE table configs, e.g. sharded table offsets, etc.
        Should be called before loading weights.
        """
        self.table_sharding_offset = table_sharding_offset

    @torch.jit.export
    def embedding_inplace_update(
        self,
        update_table_indices: List[int],
        update_row_indices: List[List[int]],
        update_weights: List[Tensor],
    ) -> None:
        # function is not used for now on the inference side
        for i in range(len(update_table_indices)):
            self.embedding_inplace_update_per_table(
                update_table_indices[i],
                torch.tensor(
                    update_row_indices[i], device=self.current_device, dtype=torch.int64
                ),
                update_weights[i],
                None,
            )

    @torch.jit.export
    def embedding_inplace_update_per_table(
        self,
        table_id: int,
        update_row_indices: Tensor,
        update_weights: Tensor,
        inplace_update_ts_sec: Optional[int] = None,
    ) -> None:
        assert table_id < len(
            self.embedding_specs
        ), f"table index {table_id} is out of range {len(self.embedding_specs)}"
        # pyre-ignore [29]
        table_offset = self.hash_size_cumsum[table_id]
        sharding_offset = self.table_sharding_offset[table_id]

        row_size = update_row_indices.numel()
        if row_size == 0:
            return

        # convert global weight index to fused local weight index
        row_indices = update_row_indices + table_offset - sharding_offset
        # set weight by id
        self.kv_embedding_cache.set_embeddings(
            row_indices, update_weights, inplace_update_ts_sec
        )

    @torch.jit.export
    def log_inplace_update_stats(
        self,
    ) -> None:
        self.kv_embedding_cache.log_inplace_update_stats()

    @torch.jit.export
    def embedding_trigger_evict(
        self,
        inplace_update_ts_sec: int,
    ) -> None:
        self.kv_embedding_cache.trigger_evict(inplace_update_ts_sec)

    @torch.jit.export
    def embedding_wait_evict_completion(
        self,
    ) -> None:
        self.kv_embedding_cache.wait_evict_completion()

    @torch.jit.export
    def initialize_kv_embedding_cache(self) -> None:
        if not self.kv_embedding_cache_initialized:
            self.initialize_logical_weights_placements_and_offsets()

            self.row_alignment = (
                8 if self.use_cpu else self.row_alignment
            )  # in order to use mempool implementation for kv embedding it needs to be divisible by 8

            hash_size_cumsum = self.construct_hash_size_cumsum()
            self.hash_size_cumsum = torch.tensor(
                hash_size_cumsum,
                dtype=torch.int64,
                device=self.current_device,
            )

            self.feature_hash_size_cumsum = torch.tensor(
                [hash_size_cumsum[t] for t in self.feature_table_map]
                + [hash_size_cumsum[-1]],
                dtype=torch.int64,
                device=self.current_device,
            )

            self.kv_embedding_cache.init(
                self.specs,
                self.row_alignment,
                self.scale_bias_size_in_bytes,
                self.hash_size_cumsum,
            )
            self.kv_embedding_cache_initialized = True
