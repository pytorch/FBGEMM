#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch  # usort:skip
from torch import Tensor  # usort:skip
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    inputs_to_device,
    rounded_row_size_in_bytes,
)
from fbgemm_gpu.tbe.cache.cache_config import CacheAlgorithm
from fbgemm_gpu.tbe.cache.kv_embedding_ops_inference import KVEmbeddingInference
from fbgemm_gpu.tbe.config import (
    BoundsCheckMode,
    DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    EmbeddingLocation,
    PoolingMode,
)

KV_CUDA_ROW_ALIGNMENT = 16
# Every refusal of a sparse delta contains this text. The module carries it as
# its `delta_refusal_marker` attribute, which the predictor matches refusals
# against, so it can change without breaking that match.
KV_SHARD_DELTA_REFUSAL_MARKER = "outside this shard's rows"
# Default of fill_new_embeddings_enabled for modules that do not set it.
KV_FILL_ROLLOUT_ENABLED = False


class KVZCHGPUBucketWiseEmbeddingInference(KVEmbeddingInference):
    """KV TBE selected by the explicit KV_ZCH_GPU_BUCKET_WISE declaration.

    The constructor's parameters, the CPU branch of ``forward`` and
    ``embedding_inplace_update_per_table`` copy the parent's (TorchScript has no
    ``super()``, and the parent's artifacts must not change): keep them in sync
    with ``KVEmbeddingInference``. ``test_cpu_forward_matches_parent`` checks
    the forward.
    """

    __constants__ = KVEmbeddingInference.__constants__ + [
        "kv_cuda_row_alignment",
        "fill_new_embeddings_supported",
    ]

    def __init__(  # noqa C901
        self,
        embedding_specs: list[
            tuple[str, int, int, SparseType, EmbeddingLocation]
        ],  # tuple of (feature_names, rows, dims, SparseType, EmbeddingLocation/placement)
        feature_table_map: list[int] | None = None,  # [T]
        index_remapping: list[Tensor] | None = None,
        pooling_mode: PoolingMode = PoolingMode.SUM,
        device: str | int | torch.device | None = None,
        bounds_check_mode: BoundsCheckMode = BoundsCheckMode.WARNING,
        weight_lists: list[tuple[Tensor, Tensor | None]] | None = None,
        pruning_hash_load_factor: float = 0.5,
        use_array_for_index_remapping: bool = True,
        output_dtype: SparseType = SparseType.FP16,
        cache_algorithm: CacheAlgorithm = CacheAlgorithm.LRU,
        cache_load_factor: float = 0.2,
        cache_sets: int = 0,
        cache_reserved_memory: float = 0.0,
        enforce_hbm: bool = False,  # place all weights/momentums in HBM when using cache
        record_cache_metrics: object | None = None,
        gather_uvm_cache_stats: bool | None = False,
        row_alignment: int | None = None,
        fp8_exponent_bits: int | None = None,
        fp8_exponent_bias: int | None = None,
        cache_assoc: int = 32,
        scale_bias_size_in_bytes: int = DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
        cacheline_alignment: bool = True,
        uvm_host_mapped: bool = False,  # True to use cudaHostAlloc; False to use cudaMallocManaged.
        reverse_qparam: bool = False,  # True to load qparams at end of each row; False to load qparam at begnning of each row.
        feature_names_per_table: list[list[str]] | None = None,
        indices_dtype: torch.dtype = torch.int32,  # Used for construction of the remap_indices tensors.  Should match the dtype of the indices passed in the forward() call (INT32 or INT64).
        embedding_cache_mode: bool = False,  # True for zero initialization, False for randomized initialization
        # 0: INT4 rows are rowwise (fp16 scale and bias). Otherwise every table is
        # INT4 stored as group-wise INT4 with one float8_e4m3fn scale per this many
        # dims (fbgemm.FloatOrHalfToGroupedInt4E4M3); lookups decode the rows to
        # FP16, which the kernel then reads as FP16 tables.
        grouped_int4_group_size: int = 0,
        # Compiles calls to the native fill_new_embeddings into the artifact,
        # which predictors without it cannot load; publish sets it per table
        # once every tenant serving the table runs such a predictor. Without
        # it, fill-only deltas are refused. None takes KV_FILL_ROLLOUT_ENABLED.
        fill_new_embeddings_enabled: bool | None = None,
    ) -> None:  # noqa C901  # tuple of (rows, dims,)
        super().__init__(
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
            embedding_cache_mode=embedding_cache_mode,
            grouped_int4_group_size=grouped_int4_group_size,
        )
        self.kv_cuda_row_alignment: int = KV_CUDA_ROW_ALIGNMENT
        self.kv_cuda_uniform_rows: bool = False
        self.delta_refusal_marker: str = KV_SHARD_DELTA_REFUSAL_MARKER
        if fill_new_embeddings_enabled is None:
            fill_new_embeddings_enabled = KV_FILL_ROLLOUT_ENABLED
        self.fill_new_embeddings_supported: bool = (
            fill_new_embeddings_enabled
            and hasattr(self.kv_embedding_cache, "fill_new_embeddings")
        )
        # Grouped INT4 rows are stored, and cached in HBM, as opaque bytes.
        self.kv_cuda_table_row_bytes: list[int] = [
            (
                (self.grouped_int4_row_bytes + KV_CUDA_ROW_ALIGNMENT - 1)
                // KV_CUDA_ROW_ALIGNMENT
                * KV_CUDA_ROW_ALIGNMENT
                if self.use_grouped_int4_rows
                else rounded_row_size_in_bytes(
                    dims, sparse_type, KV_CUDA_ROW_ALIGNMENT, scale_bias_size_in_bytes
                )
            )
            for (_, _, dims, sparse_type, _) in self.embedding_specs
        ]
        # True when every table shares one dim and weight type and rows are not
        # pooled, so CUDA lookups can treat all indices as one table.
        self.kv_cuda_single_spec: bool = (
            not self.use_grouped_int4_rows
            and int(self.pooling_mode) == int(PoolingMode.NONE)
            and len(
                {
                    (dims, sparse_type)
                    for (_, _, dims, sparse_type, _) in self.embedding_specs
                }
            )
            == 1
        )
        self.kv_cuda_row_dim: int = (
            self.embedding_specs[0][2] if self.embedding_specs else 0
        )

    @torch.jit.export
    def set_device(self, device_str: str) -> None:
        """
        Sets the device of this module. GPU lowering also treats modules with
        ``set_device`` as leaves, which keeps the KV lookup out of lowered code.
        """
        self.current_device = torch.device(device_str)

    def lookup_kv_rows(
        self,
        rows: Tensor,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Tensor | None,
        decoded_grouped_int4: bool = False,
    ) -> Tensor:
        """
        Pools ``rows``, which hold one CUDA-aligned row per index in index order.
        Uses only spec-derived buffers: modules built for publish have no weight
        placements until the predictor loads them. With
        ``decoded_grouped_int4``, ``rows`` are grouped INT4 rows already decoded
        to FP16.
        """
        D_offsets = self.D_offsets
        weights_tys = self.weights_tys
        max_int4_D = self.max_int4_D
        max_float16_D = self.max_float16_D
        if decoded_grouped_int4:
            weights_tys = torch.full_like(weights_tys, self.decoded_weight_ty)
            max_int4_D = 0
            max_float16_D = self.grouped_int4_dim
        total_D = self.total_D
        if self.kv_cuda_single_spec:
            # The kernel runs one bag per warp, and offsets give one bag per
            # feature. Unpooled rows of one spec dequantize the same as a single
            # table, so one bag per index spreads them over all warps.
            D_offsets = D_offsets[:2]
            weights_tys = weights_tys[:1]
            total_D = self.kv_cuda_row_dim
            offsets = torch.arange(
                indices.numel() + 1, device=indices.device, dtype=offsets.dtype
            )
        # Helper tensors derive from the inputs rather than being created on a
        # device, so they stay together under the dispatch modes lowering uses.
        feature_zeros = torch.zeros_like(D_offsets[1:])
        return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
            dev_weights=rows.view(-1),
            uvm_weights=rows.view(-1)[:0],
            # EmbeddingLocation.DEVICE (0) on CUDA, HOST (3) for the CPU kernel.
            weights_placements=(feature_zeros + (0 if indices.is_cuda else 3)).to(
                torch.int32
            ),
            weights_offsets=feature_zeros.to(torch.int64),
            weights_tys=weights_tys,
            D_offsets=D_offsets,
            total_D=total_D,
            max_int2_D=self.max_int2_D,
            max_int4_D=max_int4_D,
            max_int8_D=self.max_int8_D,
            max_float16_D=max_float16_D,
            max_float32_D=self.max_float32_D,
            indices=torch.arange(
                indices.numel(), device=indices.device, dtype=indices.dtype
            ),
            offsets=offsets,
            pooling_mode=int(self.pooling_mode),
            indice_weights=per_sample_weights,
            output_dtype=self.output_dtype,
            lxu_cache_weights=rows[:0],
            lxu_cache_locations=torch.zeros_like(indices[:0], dtype=torch.int32),
            row_alignment=self.kv_cuda_row_alignment,
            max_float8_D=self.max_float8_D,
            fp8_exponent_bits=self.fp8_exponent_bits,
            fp8_exponent_bias=self.fp8_exponent_bias,
        )

    def decode_grouped_int4_rows(self, rows: Tensor) -> Tensor:
        """Decodes grouped INT4 rows, e.g. from the HBM row cache, to FP16 rows
        on their device."""
        return torch.ops.fbgemm.GroupedInt4E4M3ToHalf(
            rows,
            self.grouped_int4_group_size,
            self.grouped_int4_dim,
            as_bytes=True,
        )

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Tensor | None = None,
    ) -> Tensor:
        indices, offsets, per_sample_weights = inputs_to_device(
            indices, offsets, per_sample_weights, self.bounds_check_warning
        )

        assert (
            self.weight_initialized
        ), "weight needs to be initialized before forward function"

        if indices.is_cuda:
            assert (
                self.kv_embedding_cache_initialized
            ), "KV CUDA lookup requires initialize_kv_embedding_cache first"
            assert (
                self.kv_cuda_uniform_rows
            ), "KV CUDA lookup requires one CUDA-aligned row width across tables"
            linear_indices = torch.ops.fbgemm.linearize_cache_indices(
                self.feature_hash_size_cumsum.to(indices.device),
                indices,
                offsets,
            )
            rows = self.kv_embedding_cache.get_embeddings(linear_indices)
            if self.use_grouped_int4_rows:
                # Grouped INT4 rows, e.g. from the HBM row cache, decode on
                # the GPU to the FP16 rows the lookup kernel pools.
                return self.lookup_kv_rows(
                    self.decode_grouped_int4_rows(rows),
                    indices,
                    offsets,
                    per_sample_weights,
                    decoded_grouped_int4=True,
                )
            return self.lookup_kv_rows(rows, indices, offsets, per_sample_weights)

        lxu_cache_locations = self.lxu_cache_locations_list.pop()

        weights_offsets = self.weights_offsets
        weights = self.weights_host if self.host_size > 0 else self.weights_dev
        weights_tys = self.weights_tys
        max_int4_D = self.max_int4_D
        max_float16_D = self.max_float16_D

        if self.kv_embedding_cache_initialized:
            indices = self.linearize_cache_indices(
                indices,
                offsets,
            )

            weights = self.kv_embedding_cache.get_embeddings(indices)
            if self.use_grouped_int4_rows:
                weights = torch.ops.fbgemm.GroupedInt4E4M3ToHalf(
                    weights,
                    self.grouped_int4_group_size,
                    self.grouped_int4_dim,
                    as_bytes=True,
                )
                # Derived from weights_tys, which serving rebuilds on the target
                # device (recompute_module_buffers); a buffer would stay on meta.
                weights_tys = torch.full_like(weights_tys, self.decoded_weight_ty)
                max_int4_D = 0
                max_float16_D = self.grouped_int4_dim

            indices, weights_offsets = self.calculate_indices_and_weights_offsets(
                indices, offsets
            )

        return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
            dev_weights=weights,
            uvm_weights=self.weights_uvm,
            weights_placements=self.weights_placements,
            weights_offsets=weights_offsets,
            weights_tys=weights_tys,
            D_offsets=self.D_offsets,
            total_D=self.total_D,
            max_int2_D=self.max_int2_D,
            max_int4_D=max_int4_D,
            max_int8_D=self.max_int8_D,
            max_float16_D=max_float16_D,
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

    @torch.jit.export
    def embedding_inplace_update_per_table(
        self,
        table_id: int,
        update_row_indices: Tensor,
        update_weights: Tensor,
        inplace_update_ts_sec: int | None = None,
    ) -> None:
        assert table_id < len(
            self.embedding_specs
        ), f"table index {table_id} is out of range {len(self.embedding_specs)}"
        table_offset = self.hash_size_cumsum[table_id]
        sharding_offset = self.table_sharding_offset[table_id]

        row_size = update_row_indices.numel()
        if update_weights.dim() != 2 or update_weights.size(0) != row_size:
            raise RuntimeError(
                f"KV-ZCH GPU shard update needs one row per index: {row_size} "
                f"indices, weights of shape {list(update_weights.shape)}"
            )
        if row_size == 0:
            return
        self._check_grouped_int4_rows(update_weights)
        if inplace_update_ts_sec is None:
            # A sparse delta (snapshot loads pass their timestamp) can name one
            # shard's weight with ids of every shard: writing them here would
            # store other shards' ids as keys this shard never serves. Dropping
            # them instead would lose those rows, since only fill-only deltas
            # (embedding_fill_new_rows_per_table) are fanned out to every
            # shard. So a delta of a multi-shard GPU-served table with mutable
            # values is refused whole, and the predictor counts the refusal by
            # delta_refusal_marker.
            local_rows = update_row_indices.reshape(-1) - sharding_offset
            if not bool(
                (
                    (local_rows >= 0) & (local_rows < self.embedding_specs[table_id][1])
                ).all()
            ):
                raise RuntimeError(
                    f"KV-ZCH GPU shard delta has ids {self.delta_refusal_marker} "
                    f"[{int(sharding_offset)}, "
                    f"{int(sharding_offset) + self.embedding_specs[table_id][1]}); "
                    "refusing it"
                )

        # convert global weight index to fused local weight index
        row_indices = (
            update_row_indices + table_offset.to(update_row_indices.device)
        ) - sharding_offset
        # set weight by id
        self.kv_embedding_cache.set_embeddings(
            row_indices, update_weights, inplace_update_ts_sec
        )

    def _check_grouped_int4_rows(self, update_weights: Tensor) -> None:
        if self.use_grouped_int4_rows and not (
            update_weights.dtype == torch.uint8
            and update_weights.dim() == 2
            and update_weights.size(1) == self.grouped_int4_row_bytes
        ):
            # The store zero-pads narrower rows, so a rowwise INT4 row written
            # here would be silently misdecoded instead of rejected.
            raise ValueError(
                f"grouped INT4 rows must be uint8 with {self.grouped_int4_row_bytes} "
                f"bytes, got {update_weights.dtype} {list(update_weights.shape)}"
            )

    @torch.jit.export
    def embedding_fill_new_rows_per_table(
        self,
        table_id: int,
        update_row_indices: Tensor,
        update_weights: Tensor,
    ) -> None:
        """
        Writes the rows of ids this row-wise shard has no row for to the KV
        store only, never to an HBM row cache: fill-only sparse deltas, e.g.
        RES, that never change an existing row. Ids are global; rows outside
        [table_sharding_offset, table_sharding_offset + rows) belong to another
        shard and are dropped. Kept ids are stored under the same key as the
        snapshot load.
        """
        if not self.fill_new_embeddings_supported:
            raise RuntimeError("KV fill-only deltas require native fill_new_embeddings")
        else:
            assert table_id < len(
                self.embedding_specs
            ), f"table index {table_id} is out of range {len(self.embedding_specs)}"
            if update_row_indices.numel() == 0:
                return
            self._check_grouped_int4_rows(update_weights)
            table_offset = self.hash_size_cumsum[table_id]
            local_rows = update_row_indices - self.table_sharding_offset[table_id]
            in_shard = (local_rows >= 0) & (
                local_rows < self.embedding_specs[table_id][1]
            )
            num_in_shard = int(in_shard.sum().item())
            if num_in_shard == 0:
                return
            if num_in_shard < update_row_indices.numel():
                local_rows = local_rows[in_shard]
                update_weights = update_weights[in_shard.to(update_weights.device)]
            keys = local_rows + table_offset.to(local_rows.device)
            self.kv_embedding_cache.fill_new_embeddings(keys, update_weights)

    @torch.jit.export
    def initialize_kv_embedding_cache(self) -> None:
        if not self.kv_embedding_cache_initialized:
            self.row_alignment = (
                self.kv_int4_logical_row_alignment
                if self.use_packed_int4_rows
                else self.kv_storage_row_alignment
            )
            self.initialize_logical_weights_placements_and_offsets()

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

            if self.use_packed_int4_rows:
                self.kv_embedding_cache.init_with_row_alignments(
                    self.specs,
                    self.kv_int4_logical_row_alignment,
                    self.kv_storage_row_alignment,
                    self.scale_bias_size_in_bytes,
                    self.hash_size_cumsum,
                )
            else:
                self.kv_embedding_cache.init(
                    self.specs,
                    self.kv_storage_row_alignment,
                    self.scale_bias_size_in_bytes,
                    self.hash_size_cumsum,
                )
            cuda_row_bytes = self.kv_embedding_cache.get_cuda_row_bytes()
            uniform_rows = True
            for row_bytes in self.kv_cuda_table_row_bytes:
                uniform_rows = uniform_rows and row_bytes == cuda_row_bytes
            self.kv_cuda_uniform_rows = uniform_rows
            self.kv_embedding_cache_initialized = True
