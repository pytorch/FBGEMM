#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import itertools
import logging
import os
import tempfile
from math import log2
from typing import List, Optional, Tuple

import torch  # usort:skip

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    align_to_cacheline,
    rounded_row_size_in_bytes,
    unpadded_row_size_in_bytes,
)

from torch import distributed as dist, nn, Tensor  # usort:skip
from torch.autograd.profiler import record_function

from .common import ASSOC


class SSDIntNBitTableBatchedEmbeddingBags(nn.Module):
    """
    SSD Table-batched version of nn.EmbeddingBag(sparse=False)
    Inference version, with FP32/FP16/FP8/INT8/INT4/INT2 supports
    """

    embedding_specs: List[Tuple[str, int, int, SparseType]]
    _local_instance_index: int = -1

    def __init__(
        self,
        embedding_specs: List[
            Tuple[str, int, int, SparseType]
        ],  # tuple of (feature_names, rows, dims, SparseType)
        feature_table_map: Optional[List[int]] = None,  # [T]
        pooling_mode: PoolingMode = PoolingMode.SUM,
        output_dtype: SparseType = SparseType.FP16,
        row_alignment: Optional[int] = None,
        fp8_exponent_bits: Optional[int] = None,
        fp8_exponent_bias: Optional[int] = None,
        cache_assoc: int = 32,
        scale_bias_size_in_bytes: int = DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
        cache_sets: int = 0,
        ssd_storage_directory: str = "/tmp",
        ssd_shards: int = 1,
        ssd_memtable_flush_period: int = -1,
        ssd_memtable_flush_offset: int = -1,
        ssd_l0_files_per_compact: int = 4,
        ssd_rate_limit_mbps: int = 0,
        ssd_size_ratio: int = 10,
        ssd_compaction_trigger: int = 8,
        ssd_write_buffer_size: int = 2 * 1024 * 1024 * 1024,
        ssd_max_write_buffer_num: int = 16,
        ssd_cache_location: EmbeddingLocation = EmbeddingLocation.MANAGED,
        ssd_uniform_init_lower: float = -0.01,
        ssd_uniform_init_upper: float = 0.01,
        # Parameter Server Configs
        ps_hosts: Optional[Tuple[Tuple[str, int]]] = None,
        ps_max_key_per_request: Optional[int] = None,
        ps_client_thread_num: Optional[int] = None,
        ps_max_local_index_length: Optional[int] = None,
        tbe_unique_id: int = -1,  # unique id for this embedding, if not set, will derive based on current rank and tbe index id
    ) -> None:  # noqa C901  # tuple of (rows, dims,)
        super(SSDIntNBitTableBatchedEmbeddingBags, self).__init__()

        assert cache_assoc == 32, "Only 32-way cache is supported now"

        self.scale_bias_size_in_bytes = scale_bias_size_in_bytes
        self.pooling_mode = pooling_mode
        self.embedding_specs = embedding_specs
        T_ = len(self.embedding_specs)
        assert T_ > 0
        device = torch.cuda.current_device()
        if device is None:
            self.current_device: torch.device = torch.device(
                torch.cuda.current_device()
            )
        elif isinstance(device, torch.device):
            self.current_device = device
        else:
            self.current_device = torch.device(device)
        self.use_cpu: bool = self.current_device.type == "cpu"

        self.feature_table_map: List[int] = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        T = len(self.feature_table_map)
        assert T_ <= T
        table_has_feature = [False] * T_
        for t in self.feature_table_map:
            table_has_feature[t] = True
        assert all(table_has_feature), "Each table must have at least one feature!"

        self.output_dtype: int = output_dtype.as_int()
        # (feature_names, rows, dims, weights_tys) = zip(*embedding_specs)
        # Pyre workaround
        rows: List[int] = [e[1] for e in embedding_specs]
        dims: List[int] = [e[2] for e in embedding_specs]
        weights_tys: List[SparseType] = [e[3] for e in embedding_specs]

        D_offsets = [dims[t] for t in self.feature_table_map]
        D_offsets = [0] + list(itertools.accumulate(D_offsets))
        self.total_D: int = D_offsets[-1]
        self.register_buffer(
            "D_offsets",
            torch.tensor(D_offsets, device=self.current_device, dtype=torch.int32),
        )

        if row_alignment is None:
            self.row_alignment: int = 1 if self.use_cpu else 16
        else:
            self.row_alignment = row_alignment

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

        cached_dims = [
            rounded_row_size_in_bytes(
                embedding_spec[2], embedding_spec[3], 16, self.scale_bias_size_in_bytes
            )
            for embedding_spec in self.embedding_specs
        ]
        self.max_D_cache: int = max(cached_dims) if len(cached_dims) > 0 else 0

        placements = []
        offsets = []
        uvm_size = 0
        for _, num_embeddings, embedding_dim, weight_ty in embedding_specs:
            embedding_dim = rounded_row_size_in_bytes(
                embedding_dim, weight_ty, self.row_alignment, scale_bias_size_in_bytes
            )
            state_size = num_embeddings * embedding_dim
            state_size = align_to_cacheline(state_size)
            placements.append(EmbeddingLocation.MANAGED_CACHING)
            offsets.append(uvm_size)
            uvm_size += state_size

        self.weights_physical_offsets: List[int] = offsets

        weights_tys_int = [weights_tys[t].as_int() for t in self.feature_table_map]
        self.register_buffer(
            "weights_tys",
            torch.tensor(
                weights_tys_int, device=self.current_device, dtype=torch.uint8
            ),
        )
        self.weight_initialized: bool = True

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
        assert cache_sets > 0
        element_size = 1
        cache_size = cache_sets * ASSOC * element_size * self.max_D_cache
        logging.info(
            f"Using cache for SSD with admission algorithm "
            f"{CacheAlgorithm.LRU}, {cache_sets} sets, stored on {'DEVICE' if ssd_cache_location is EmbeddingLocation.DEVICE else 'MANAGED'} with {ssd_shards} shards, "
            f"SSD storage directory: {ssd_storage_directory}, "
            f"Memtable Flush Period: {ssd_memtable_flush_period}, "
            f"Memtable Flush Offset: {ssd_memtable_flush_offset}, "
            f"Desired L0 files per compaction: {ssd_l0_files_per_compact}, "
            f"{cache_size / 1024.0 / 1024.0 / 1024.0 : .2f}GB, "
            f"output dtype: {output_dtype}"
        )
        self.register_buffer(
            "lxu_cache_state",
            torch.zeros(cache_sets, ASSOC, dtype=torch.int64).fill_(-1),
        )
        self.register_buffer(
            "lru_state", torch.zeros(cache_sets, ASSOC, dtype=torch.int64)
        )

        assert ssd_cache_location in (
            EmbeddingLocation.MANAGED,
            EmbeddingLocation.DEVICE,
        )
        if ssd_cache_location == EmbeddingLocation.MANAGED:
            self.register_buffer(
                "lxu_cache_weights",
                torch.ops.fbgemm.new_managed_tensor(
                    torch.zeros(1, device=self.current_device, dtype=torch.uint8),
                    [cache_sets * ASSOC, self.max_D_cache],
                ),
            )
        else:
            self.register_buffer(
                "lxu_cache_weights",
                torch.zeros(
                    cache_sets * ASSOC,
                    self.max_D_cache,
                    device=self.current_device,
                    dtype=torch.uint8,
                ),
            )

            assert (
                cache_size
                == self.lxu_cache_weights.numel()
                * self.lxu_cache_weights.element_size()
            ), "The precomputed cache_size does not match the actual cache size"

        os.makedirs(ssd_storage_directory, exist_ok=True)

        ssd_directory = tempfile.mkdtemp(
            prefix="ssd_table_batched_embeddings", dir=ssd_storage_directory
        )
        if not ps_hosts:
            # pyre-fixme[4]: Attribute must be annotated.
            # pyre-ignore[16]
            self.ssd_db = torch.classes.fbgemm.EmbeddingRocksDBWrapper(
                ssd_directory,
                ssd_shards,
                ssd_shards,
                ssd_memtable_flush_period,
                ssd_memtable_flush_offset,
                ssd_l0_files_per_compact,
                self.max_D_cache,
                ssd_rate_limit_mbps,
                ssd_size_ratio,
                ssd_compaction_trigger,
                ssd_write_buffer_size,
                ssd_max_write_buffer_num,
                ssd_uniform_init_lower,
                ssd_uniform_init_upper,
                8,  # row_storage_bitwidth
                0,  # ssd_block_cache_size
            )
        else:
            # create tbe unique id using rank index | pooling mode
            if tbe_unique_id == -1:
                SSDIntNBitTableBatchedEmbeddingBags._local_instance_index += 1
                assert (
                    SSDIntNBitTableBatchedEmbeddingBags._local_instance_index < 8
                ), f"{SSDIntNBitTableBatchedEmbeddingBags._local_instance_index}, more than 8 TBE instance  is created in one rank, the tbe unique id won't be unique in this case."
                tbe_unique_id = (
                    dist.get_rank() << 3
                    | SSDIntNBitTableBatchedEmbeddingBags._local_instance_index
                )
            logging.info(f"tbe_unique_id: {tbe_unique_id}")
            # pyre-fixme[4]: Attribute must be annotated.
            # pyre-ignore[16]
            self.ssd_db = torch.classes.fbgemm.EmbeddingParameterServerWrapper(
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
                0,  # ssd_block_cache_size
                self.max_D_cache,
            )

        # pyre-fixme[20]: Argument `self` expected.
        (low_priority, high_priority) = torch.cuda.Stream.priority_range()
        self.ssd_stream = torch.cuda.Stream(priority=low_priority)
        self.ssd_set_start = torch.cuda.Event()
        self.ssd_set_end = torch.cuda.Event()

        # pyre-fixme[4]: Attribute must be annotated.
        # pyre-ignore[16]
        self.timestep_counter = torch.classes.fbgemm.AtomicCounter()
        # pyre-fixme[4]: Attribute must be annotated.
        # pyre-ignore[16]
        self.timestep_prefetch_size = torch.classes.fbgemm.AtomicCounter()

        self.weights_dev: torch.Tensor = torch.empty(
            0,
            device=self.current_device,
            dtype=torch.uint8,
        )
        self.register_buffer(
            "weights_uvm",
            torch.tensor((0,), device=self.current_device, dtype=torch.uint8),
        )
        self.register_buffer(
            "weights_host",
            torch.empty(0),
        )

        self.register_buffer(
            "weights_placements",
            torch.tensor(
                [EmbeddingLocation.MANAGED_CACHING for _ in range(T_)],
                dtype=torch.int32,
            ),
        )
        weights_offsets = [0] + list(
            itertools.accumulate([row * dim for (row, dim) in zip(rows, dims)])
        )
        self.register_buffer(
            "weights_offsets",
            torch.tensor(
                weights_offsets[:-1],
                device=self.current_device,
                dtype=torch.int64,
            ),
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

    @torch.jit.export
    def prefetch(self, indices: Tensor, offsets: Tensor) -> Tensor:
        (indices, offsets) = indices.long(), offsets.long()
        linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
            self.hash_size_cumsum,
            indices,
            offsets,
        )
        self.timestep_counter.increment()
        self.timestep_prefetch_size.increment()
        (
            inserted_indices,
            evicted_indices,
            assigned_cache_slots,
            actions_count_gpu,
            _,
            _,
            _,
            _,
        ) = torch.ops.fbgemm.ssd_cache_populate_actions(
            linear_cache_indices,
            self.total_hash_size,
            self.lxu_cache_state,
            self.timestep_counter.get(),
            1,  # for now assume prefetch_dist == 1
            self.lru_state,
        )
        actions_count_cpu = torch.empty(
            actions_count_gpu.shape, pin_memory=True, dtype=actions_count_gpu.dtype
        )
        actions_count_cpu.copy_(actions_count_gpu, non_blocking=True)
        assigned_cache_slots = assigned_cache_slots.long()
        evicted_rows = self.lxu_cache_weights[
            assigned_cache_slots.clamp_(min=0).long(), :
        ]
        inserted_rows = torch.empty(
            evicted_rows.shape,
            dtype=self.lxu_cache_weights.dtype,
            pin_memory=True,
        )

        current_stream = torch.cuda.current_stream()

        # Ensure the previous iterations l3_db.set(..) has completed.
        current_stream.wait_event(self.ssd_set_end)
        inserted_indices_cpu = torch.empty(
            inserted_indices.shape, pin_memory=True, dtype=inserted_indices.dtype
        )
        inserted_indices_cpu.copy_(inserted_indices, non_blocking=True)
        self.ssd_db.get_cuda(
            inserted_indices_cpu,
            inserted_rows,
            actions_count_cpu,
        )
        current_stream.record_event(self.ssd_set_start)
        # TODO: T123943415 T123943414 this is a big copy that is (mostly) unnecessary with a decent cache hit rate.
        # Should we allocate on HBM?
        inserted_rows_gpu = inserted_rows.to(self.current_device, non_blocking=True)

        # self.lxu_cache_weights[assigned_cache_slots, :] = inserted_rows.cuda(non_blocking=True)
        torch.ops.fbgemm.masked_index_put(
            self.lxu_cache_weights,
            assigned_cache_slots,
            inserted_rows_gpu,
            actions_count_gpu,
        )

        with torch.cuda.stream(self.ssd_stream):
            self.ssd_stream.wait_event(self.ssd_set_start)
            evicted_rows_cpu = torch.empty(
                evicted_rows.shape, pin_memory=True, dtype=evicted_rows.dtype
            )
            evicted_rows_cpu.copy_(evicted_rows, non_blocking=True)
            evicted_indices_cpu = torch.empty(
                evicted_indices.shape, pin_memory=True, dtype=evicted_indices.dtype
            )
            evicted_indices_cpu.copy_(evicted_indices, non_blocking=True)
            evicted_rows.record_stream(self.ssd_stream)
            evicted_indices.record_stream(self.ssd_stream)
            self.ssd_db.set_cuda(
                evicted_indices_cpu,
                evicted_rows_cpu,
                actions_count_cpu,
                self.timestep_counter.get(),
            )
            # TODO: is this needed?
            # Need a way to synchronize
            #  actions_count_cpu.record_stream(self.ssd_stream)
            self.ssd_stream.record_event(self.ssd_set_end)
        return linear_cache_indices

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        if self.timestep_prefetch_size.get() <= 0:
            with record_function("## prefetch ##"):
                linear_cache_indices = self.prefetch(indices, offsets)
        else:
            linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
                self.hash_size_cumsum,
                indices,
                offsets,
            )
        lxu_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
            linear_cache_indices,
            self.lxu_cache_state,
            self.total_hash_size,
        )

        self.timestep_prefetch_size.decrement()

        assert (
            self.weight_initialized
        ), "weight needs to be initialized before forward function"

        # Note: CPU and CUDA ops use the same interface to facilitate JIT IR
        # generation for CUDA/CPU. For CPU op, we don't need weights_uvm and
        # weights_placements
        return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
            dev_weights=self.weights_dev,
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

    @torch.jit.export
    def split_embedding_weights(
        self, split_scale_shifts: bool = True
    ) -> List[Tuple[Tensor, Optional[Tensor]]]:
        """
        Returns a list of weights, split by table.

        Testing only, very slow.
        """
        splits: List[Tuple[Tensor, Optional[Tensor]]] = []
        rows_cumsum = 0
        for _, row, dim, weight_ty in self.embedding_specs:
            weights = torch.empty(
                (
                    row,
                    rounded_row_size_in_bytes(
                        dim,
                        weight_ty,
                        self.row_alignment,
                        self.scale_bias_size_in_bytes,
                    ),
                ),
                dtype=torch.uint8,
            )
            self.ssd_db.get_cuda(
                torch.arange(rows_cumsum, rows_cumsum + row).to(torch.int64),
                weights,
                torch.as_tensor([row]),
            )
            rows_cumsum += row
            torch.cuda.synchronize(self.current_device)

            weights_shifts = weights.detach()

            if split_scale_shifts:
                # remove the padding at the end of each row.
                weights_shifts = weights_shifts[
                    :,
                    : unpadded_row_size_in_bytes(
                        dim, weight_ty, self.scale_bias_size_in_bytes
                    ),
                ]
                if (
                    weight_ty == SparseType.INT8
                    or weight_ty == SparseType.INT4
                    or weight_ty == SparseType.INT2
                ):
                    splits.append(
                        (
                            weights_shifts[:, self.scale_bias_size_in_bytes :],
                            weights_shifts[:, : self.scale_bias_size_in_bytes],
                        )
                    )
                else:
                    assert (
                        weight_ty == SparseType.FP8
                        or weight_ty == SparseType.FP16
                        or weight_ty == SparseType.FP32
                    )
                    splits.append(
                        (
                            weights_shifts,
                            None,
                        )
                    )
            else:
                splits.append((weights_shifts, None))

        torch.cuda.synchronize(self.current_device)
        return splits
