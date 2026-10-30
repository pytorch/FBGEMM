#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]


import torch  # usort:skip
from torch import Tensor  # usort:skip
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    inputs_to_device,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    random_quant_scaled_tensor,
    rounded_row_size_in_bytes,
)
from fbgemm_gpu.tbe.config import (
    BoundsCheckMode,
    DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    EmbeddingLocation,
    PoolingMode,
    RecordCacheMetrics,
)
from fbgemm_gpu.utils.loader import load_torch_module

from .cache_config import CacheAlgorithm  # usort:skip

KV_STORAGE_ROW_ALIGNMENT = 8
KV_INT4_LOGICAL_ROW_ALIGNMENT = 1
# Keep disabled until the native custom-class schema is in the minimum serving
# package. Enabling it makes new TorchScript artifacts require that schema.
KV_INT4_PACKED_ROWS_ROLLOUT_ENABLED = False


def _supports_packed_int4_rows(kv_embedding_cache: object) -> bool:
    return hasattr(kv_embedding_cache, "init_with_row_alignments")


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

    __constants__ = [
        "use_packed_int4_rows",
        "use_grouped_int4_rows",
        "kv_int4_logical_row_alignment",
        "kv_storage_row_alignment",
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
        record_cache_metrics: RecordCacheMetrics | None = None,
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
    ) -> None:  # noqa C901  # tuple of (rows, dims,)
        super().__init__(
            # pyre-fixme[6]: Type-identity mismatch on EmbeddingLocation between shell
            # and canonical package; resolves once D103477971 unifies the classes via re-export.
            embedding_specs=embedding_specs,
            feature_table_map=feature_table_map,
            index_remapping=index_remapping,
            # pyre-fixme[6]: Type-identity mismatch on PoolingMode between shell and
            # canonical package; resolves once D103477971 unifies the classes via re-export.
            pooling_mode=pooling_mode,
            device=device,
            # pyre-fixme[6]: Type-identity mismatch on BoundsCheckMode between shell
            # and canonical package; resolves once D103477971 unifies the classes via re-export.
            bounds_check_mode=bounds_check_mode,
            weight_lists=weight_lists,
            pruning_hash_load_factor=pruning_hash_load_factor,
            use_array_for_index_remapping=use_array_for_index_remapping,
            output_dtype=output_dtype,
            # pyre-fixme[6]: Type-identity mismatch on CacheAlgorithm between shell
            # and canonical package; resolves once D103477971 unifies the classes via re-export.
            cache_algorithm=cache_algorithm,
            cache_load_factor=cache_load_factor,
            cache_sets=cache_sets,
            cache_reserved_memory=cache_reserved_memory,
            enforce_hbm=enforce_hbm,
            # pyre-fixme[6]: Type-identity mismatch on RecordCacheMetrics between shell
            # and canonical package; resolves once D103477971 unifies the classes via re-export.
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
        # A constant, so modules without grouped rows script without the decode op
        # and load on serving packages that predate it.
        self.use_grouped_int4_rows: bool = grouped_int4_group_size > 0
        # Plain attributes, which serving loaders read to size and check rows.
        self.grouped_int4_group_size: int = grouped_int4_group_size
        self.grouped_int4_dim: int = 0
        self.grouped_int4_row_bytes: int = 0
        if grouped_int4_group_size:
            # Forward serves decoded rows as an FP16 table; the quantized output
            # kernels would size their output for the INT4 table instead.
            assert output_dtype in (
                SparseType.FP32,
                SparseType.FP16,
                SparseType.BF16,
            ), f"grouped_int4_group_size needs a float output_dtype, got {output_dtype}"
            self.grouped_int4_dim = self._grouped_int4_dim(grouped_int4_group_size)
            self.grouped_int4_row_bytes = (
                self.grouped_int4_dim // 2
                + self.grouped_int4_dim // grouped_int4_group_size
            )
        self.decoded_weight_ty: int = SparseType.FP16.as_int()
        self.embedding_cache_mode: bool = embedding_cache_mode

        num_shards = 32
        uniform_init_lower: float = -0.01
        uniform_init_upper: float = 0.01

        # pyre-fixme[4]: Attribute must be annotated.
        self.kv_embedding_cache = torch.classes.fbgemm.DramKVEmbeddingInferenceWrapper(
            num_shards,
            uniform_init_lower,
            uniform_init_upper,
            embedding_cache_mode,  # in embedding_cache_mode, we disable random init
        )

        # The KV store sizes rows from these specs. Grouped rows are opaque bytes,
        # declared as dim / 2 + dim / group_size one-byte (FP8-sized) columns.
        self.specs: list[tuple[int, int, int]] = [
            (
                (
                    rows,
                    dims // 2 + dims // grouped_int4_group_size,
                    SparseType.FP8.as_int(),
                )
                if grouped_int4_group_size
                else (rows, dims, sparse_type.as_int())
            )
            for (_, rows, dims, sparse_type, _) in self.embedding_specs
        ]
        first_logical_row_bytes = (
            rounded_row_size_in_bytes(
                self.embedding_specs[0][2],
                self.embedding_specs[0][3],
                KV_INT4_LOGICAL_ROW_ALIGNMENT,
                scale_bias_size_in_bytes,
            )
            if self.embedding_specs
            else 0
        )
        self.use_packed_int4_rows: bool = (
            _supports_packed_int4_rows(self.kv_embedding_cache)
            and KV_INT4_PACKED_ROWS_ROLLOUT_ENABLED
            and not grouped_int4_group_size
            and bool(self.embedding_specs)
            and all(
                [
                    sparse_type == SparseType.INT4
                    and rounded_row_size_in_bytes(
                        dims,
                        sparse_type,
                        KV_INT4_LOGICAL_ROW_ALIGNMENT,
                        scale_bias_size_in_bytes,
                    )
                    == first_logical_row_bytes
                    for (_, _, dims, sparse_type, _) in self.embedding_specs
                ]
            )
        )
        self.kv_int4_logical_row_alignment: int = KV_INT4_LOGICAL_ROW_ALIGNMENT
        self.kv_storage_row_alignment: int = KV_STORAGE_ROW_ALIGNMENT
        # table shard offset if inference sharding is enabled, otherwise, should be all zeros
        self.table_sharding_offset: list[int] = [0] * len(self.embedding_specs)
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

    def _grouped_int4_dim(self, group_size: int) -> int:
        """The shared table dim, validated for group-wise INT4 storage."""
        dims = [dims for (_, _, dims, _, _) in self.embedding_specs]
        assert dims, "grouped_int4_group_size needs at least one table"
        assert all(
            sparse_type == SparseType.INT4
            for (_, _, _, sparse_type, _) in self.embedding_specs
        ), "grouped_int4_group_size needs every table to be INT4"
        assert all(d == dims[0] for d in dims), (
            f"grouped_int4_group_size needs every table to have the same dim, "
            f"got {dims}"
        )
        dim = dims[0]
        # Publish only writes group size 8 (INT4_G8). Other sizes can match a
        # padded rowwise INT4 row: group size 16 at dim 128 gives 72 bytes, a
        # rowwise row padded to 8 bytes.
        assert group_size == 8, f"grouped_int4_group_size must be 8, got {group_size}"
        assert (
            dim % group_size == 0
        ), f"grouped_int4_group_size {group_size} must divide dim {dim}"
        # grouped_int4_group_size, not the row width, identifies the format;
        # loaders and embedding_inplace_update_per_table check widths only to
        # catch rows of the other format. Keep grouped rows from matching
        # unpadded rowwise INT4 rows (dim / 2 + 4 bytes) so that check holds.
        assert dim // group_size != 4, (
            f"grouped_int4_group_size {group_size} with dim {dim} gives rows as wide "
            f"as rowwise INT4 rows"
        )
        # Decoded FP16 rows are read with the KV storage row alignment.
        assert (2 * dim) % KV_STORAGE_ROW_ALIGNMENT == 0, (
            f"grouped INT4 dim {dim} must give FP16 rows aligned to "
            f"{KV_STORAGE_ROW_ALIGNMENT} bytes"
        )
        return dim

    def construct_hash_size_cumsum(self) -> list[int]:
        hash_size_cumsum = [0]
        for spec in self.embedding_specs:
            rows = spec[1]
            hash_size_cumsum.append(hash_size_cumsum[-1] + rows)
        return hash_size_cumsum

    def calculate_indices_and_weights_offsets(
        self, indices: Tensor, offsets: Tensor
    ) -> tuple[Tensor, Tensor]:
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
            weight_ty = self.embedding_specs[table_id][3]
            if self.use_grouped_int4_rows:
                weight_ty = SparseType.FP16
            total_bytes_added += index_size * rounded_row_size_in_bytes(
                self.embedding_specs[table_id][2],  # dim
                weight_ty,
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
        per_sample_weights: Tensor | None = None,
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

    def fill_random_weights(self) -> None:
        """
        Fill the buffer with random weights, table by table
        """
        self.initialize_kv_embedding_cache()
        for i, (_, num_embeddings, embedding_dim, weight_ty, _) in enumerate(
            self.embedding_specs
        ):
            indices = torch.arange(num_embeddings, dtype=torch.int64)
            if self.use_grouped_int4_rows:
                weights = torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(
                    torch.empty(
                        num_embeddings, embedding_dim, dtype=torch.float16
                    ).uniform_(-0.01, 0.01),
                    self.grouped_int4_group_size,
                )
            else:
                weights = random_quant_scaled_tensor(
                    shape=torch.Size(
                        [
                            num_embeddings,
                            rounded_row_size_in_bytes(
                                embedding_dim, weight_ty, self.row_alignment
                            ),
                        ]
                    ),
                    device=self.current_device,
                )
            self.embedding_inplace_update_per_table(
                i,
                indices,
                weights,
            )
        self.weight_initialized = True

    @torch.jit.export
    def init_tbe_config(self, table_sharding_offset: list[int]) -> None:
        """
        Initialize the dynamic TBE table configs, e.g. sharded table offsets, etc.
        Should be called before loading weights.
        """
        self.table_sharding_offset = table_sharding_offset

    @torch.jit.export
    def embedding_inplace_update(
        self,
        update_table_indices: list[int],
        update_row_indices: list[list[int]],
        update_weights: list[Tensor],
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
        inplace_update_ts_sec: int | None = None,
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
            self.kv_embedding_cache_initialized = True
