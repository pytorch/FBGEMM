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
import threading
from collections.abc import Generator
from contextlib import contextmanager
from math import log2

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

IS_ROCM: bool = hasattr(torch.version, "hip") and torch.version.hip is not None


class _RWLock:
    """
    Lightweight read-write lock for concurrent inference + rare updates.

    Multiple readers (prefetch/forward) can proceed concurrently.
    A writer (streaming_update/load_snapshot) gets exclusive access,
    blocking new readers and waiting for in-flight readers to finish.

    Writer-priority: once any writer is waiting, new readers block until
    all waiting writers have completed. Uses a counter (not a bool) so
    that multiple queued writers don't starve each other.
    """

    def __init__(self) -> None:
        self._cond: threading.Condition = threading.Condition(threading.Lock())
        self._readers: int = 0
        self._writers_waiting: int = 0
        self._writer_active: bool = False

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writers_waiting += 1
            while self._readers > 0 or self._writer_active:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer_active = True

    def release_write(self) -> None:
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()

    @contextmanager
    def read_lock(self) -> Generator[None, None, None]:
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_lock(
        self, cuda_device: torch.device | None = None
    ) -> Generator[None, None, None]:
        self.acquire_write()
        try:
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
            yield
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
        finally:
            self.release_write()


class SSDIntNBitTableBatchedEmbeddingBags(nn.Module):
    """
    SSD Table-batched version of nn.EmbeddingBag(sparse=False)
    Inference version, with FP32/FP16/FP8/INT8/INT4/INT2 supports

    AMD/ROCm support status:
        This operator supports AMD GPUs (ROCm/HIP). Key adaptations:
        - Cache associativity (ASSOC) is set to 64 to match AMD's 64-wide
          wavefronts (vs. 32 for NVIDIA warps). Python-side tensor shapes
          and C++ kernel indexing are kept in sync via common.ASSOC.
        - BitonicSort includes a 6th merge stage (L=32) for 64-element sorts.
        - lxu_cache_lookup uses HIP-native __ballot() instead of
          __ballot_sync().
        - SM-count tuning uses CU-count-based heuristic for AMD GPUs.
    """

    embedding_specs: list[tuple[str, int, int, SparseType]]
    _local_instance_index: int = -1

    def __init__(
        self,
        embedding_specs: list[
            tuple[str, int, int, SparseType]
        ],  # tuple of (feature_names, rows, dims, SparseType)
        feature_table_map: list[int] | None = None,  # [T]
        pooling_mode: PoolingMode = PoolingMode.SUM,
        output_dtype: SparseType = SparseType.FP16,
        row_alignment: int | None = None,
        fp8_exponent_bits: int | None = None,
        fp8_exponent_bias: int | None = None,
        cache_assoc: int = ASSOC,
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
        ps_hosts: tuple[tuple[str, int]] | None = None,
        ps_max_key_per_request: int | None = None,
        ps_client_thread_num: int | None = None,
        ps_max_local_index_length: int | None = None,
        tbe_unique_id: int = -1,  # unique id for this embedding, if not set, will derive based on current rank and tbe index id
        enable_cache_locking: bool = False,  # opt-in: lock cache lines during forward to prevent eviction races
    ) -> None:  # noqa C901  # tuple of (rows, dims,)
        super().__init__()

        assert cache_assoc == ASSOC, (
            f"cache_assoc must match platform ASSOC={ASSOC} "
            f"(CUDA=32, ROCm=64), got {cache_assoc}"
        )

        self.enable_cache_locking = enable_cache_locking
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

        self.feature_table_map: list[int] = (
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
        rows: list[int] = [e[1] for e in embedding_specs]
        dims: list[int] = [e[2] for e in embedding_specs]
        weights_tys: list[SparseType] = [e[3] for e in embedding_specs]

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

        self.weights_physical_offsets: list[int] = offsets

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
        # Cache locking counter: prevents eviction of cache lines that are
        # currently being read by forward(). Without this, a concurrent
        # prefetch() can evict a cache line that forward() is still using.
        self.register_buffer(
            "lxu_cache_locking_counter",
            torch.zeros(
                cache_sets,
                ASSOC,
                device=self.current_device,
                dtype=torch.int32,
            ),
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
        self.use_ps_backend: bool = bool(ps_hosts)
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
        low_priority, high_priority = torch.cuda.Stream.priority_range()
        self.ssd_stream = torch.cuda.Stream(priority=low_priority)
        # Dedicated stream for D2H memory copies (overlaps with compute)
        self.ssd_memcpy_stream = torch.cuda.Stream(priority=low_priority)
        self.ssd_set_start = torch.cuda.Event()
        self.ssd_set_end = torch.cuda.Event()
        # Event for D2H copy completion
        self.ssd_event_memcpy = torch.cuda.Event()

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

        if IS_ROCM:
            logging.info(
                "SSD TBE inference running on ROCm with ASSOC=%d "
                "(matching AMD 64-wide wavefronts).",
                ASSOC,
            )

        # Read-write lock for concurrent inference + streaming updates.
        # prefetch/forward take shared read locks; streaming_update/load_snapshot
        # take exclusive write locks with CUDA synchronization.
        self._rw_lock: _RWLock = _RWLock()

    @torch.jit.export
    def prefetch(self, indices: Tensor, offsets: Tensor) -> Tensor:
        with self._rw_lock.read_lock():
            return self._prefetch_impl(indices, offsets)

    def _prefetch_impl(self, indices: Tensor, offsets: Tensor) -> Tensor:
        indices, offsets = indices.long(), offsets.long()
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
            lock_cache_line=self.enable_cache_locking,
            lxu_cache_locking_counter=self.lxu_cache_locking_counter,
        )
        current_stream = torch.cuda.current_stream()
        # D2H copies on dedicated memcpy stream (overlaps with compute on
        # default stream)
        actions_count_cpu = torch.empty(
            actions_count_gpu.shape, pin_memory=True, dtype=actions_count_gpu.dtype
        )
        inserted_indices_cpu = torch.empty(
            inserted_indices.shape, pin_memory=True, dtype=inserted_indices.dtype
        )
        with torch.cuda.stream(self.ssd_memcpy_stream):
            self.ssd_memcpy_stream.wait_stream(current_stream)
            actions_count_cpu.copy_(actions_count_gpu, non_blocking=True)
            inserted_indices_cpu.copy_(inserted_indices, non_blocking=True)
            self.ssd_memcpy_stream.record_event(self.ssd_event_memcpy)

        assigned_cache_slots = assigned_cache_slots.long()
        evicted_rows = self.lxu_cache_weights[
            assigned_cache_slots.clamp_(min=0).long(), :
        ]
        inserted_rows = torch.empty(
            evicted_rows.shape,
            dtype=self.lxu_cache_weights.dtype,
            pin_memory=True,
        )

        # Ensure the previous iterations l3_db.set(..) has completed.
        current_stream.wait_event(self.ssd_set_end)
        # Wait for D2H copies to complete before get_cuda
        current_stream.wait_event(self.ssd_event_memcpy)
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
        per_sample_weights: Tensor | None = None,
    ) -> Tensor:
        with self._rw_lock.read_lock():
            return self._forward_impl(indices, offsets, per_sample_weights)

    def _forward_impl(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Tensor | None = None,
    ) -> Tensor:
        if self.timestep_prefetch_size.get() <= 0:
            with record_function("## prefetch ##"):
                linear_cache_indices = self._prefetch_impl(indices, offsets)
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

        # Decrement cache locking counter — signals that these cache lines
        # are no longer pinned by this forward() call, allowing future
        # prefetch() calls to evict them if needed.
        if self.enable_cache_locking:
            torch.ops.fbgemm.lxu_cache_locking_counter_decrement(
                self.lxu_cache_locking_counter,
                lxu_cache_locations,
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
    ) -> list[tuple[Tensor, Tensor | None]]:
        """
        Returns a list of weights, split by table.

        Testing only, very slow.
        """
        splits: list[tuple[Tensor, Tensor | None]] = []
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

    @torch.jit.export
    def streaming_update(
        self,
        indices: Tensor,
        weights: Tensor,
    ) -> None:
        """
        Apply streaming embedding updates during inference.

        Writes new embedding values to RocksDB for the given indices and
        invalidates corresponding HBM cache entries so that the next
        prefetch() reloads the fresh values.

        Thread-safe: acquires an exclusive write lock and synchronizes
        all CUDA streams before and after modifying shared state.

        Args:
            indices: 1D int64 tensor of linear embedding indices to update.
            weights: 2D uint8 tensor of shape [len(indices), max_D_cache]
                     containing the new embedding bytes (including scale/bias).
        """
        assert indices.dim() == 1, "indices must be 1D"
        assert weights.dim() == 2, "weights must be 2D"
        assert indices.shape[0] == weights.shape[0], (
            f"indices and weights must have the same number of rows, "
            f"got {indices.shape[0]} and {weights.shape[0]}"
        )
        if indices.shape[0] == 0:
            return

        with self._rw_lock.write_lock(self.current_device):
            # 1. Write updated embeddings to RocksDB.
            count = torch.tensor([indices.shape[0]], dtype=torch.int64)
            self.ssd_db.set(indices.cpu(), weights.cpu(), count)

            # 2. Invalidate HBM cache entries for updated indices.
            self._invalidate_cache(indices)

    def _invalidate_cache(self, indices: Tensor) -> None:
        """Invalidate HBM cache entries for the given linear indices."""
        indices = indices.to(self.current_device)
        max_cache_sets = self.lxu_cache_state.shape[0]
        cache_set_ids = indices % max_cache_sets  # [N]

        # Gather the ASSOC slots for each relevant cache set: [N, ASSOC]
        relevant_states = self.lxu_cache_state[cache_set_ids]

        # Find which slots hold the updated indices: [N, ASSOC] bool
        matches = relevant_states == indices.unsqueeze(1)

        if matches.any():
            # Get (row_in_batch, slot) pairs for all matches
            n_idx, slot_idx = matches.nonzero(as_tuple=True)
            # Convert to flat indices into lxu_cache_state
            flat_idx = cache_set_ids[n_idx] * ASSOC + slot_idx
            self.lxu_cache_state.view(-1)[flat_idx] = -1

    @torch.jit.export
    def load_snapshot(
        self,
        ssd_storage_directory: str,
        ssd_shards: int = 1,
        ssd_memtable_flush_period: int = -1,
        ssd_memtable_flush_offset: int = -1,
        ssd_l0_files_per_compact: int = 4,
        ssd_rate_limit_mbps: int = 0,
        ssd_size_ratio: int = 10,
        ssd_compaction_trigger: int = 8,
        ssd_write_buffer_size: int = 2 * 1024 * 1024 * 1024,
        ssd_max_write_buffer_num: int = 16,
        ssd_uniform_init_lower: float = -0.01,
        ssd_uniform_init_upper: float = 0.01,
    ) -> None:
        """
        Swap to a new RocksDB snapshot without downtime.

        Opens a new EmbeddingRocksDB at the given directory and replaces
        the current ssd_db. Invalidates the entire HBM cache so that
        subsequent prefetch() calls reload from the new snapshot.

        Args:
            ssd_storage_directory: Path to the new RocksDB snapshot.
            Other args: RocksDB configuration (same as constructor).

        Raises:
            RuntimeError: If the instance uses a parameter server backend
                (ps_hosts was provided at construction time).
        """
        if self.use_ps_backend:
            raise RuntimeError(
                "load_snapshot() is only supported with the local RocksDB backend. "
                "This instance was initialized with ps_hosts (parameter server)."
            )

        with self._rw_lock.write_lock(self.current_device):
            # 1. Flush any pending writes to the old DB.
            self.ssd_db.flush()

            # 2. Open a new RocksDB instance at the new snapshot path.
            # pyre-ignore[16]
            self.ssd_db = torch.classes.fbgemm.EmbeddingRocksDBWrapper(
                ssd_storage_directory,
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

            # 3. Invalidate entire HBM cache — all entries are from old snapshot.
            self.lxu_cache_state.fill_(-1)

            logging.info(
                f"Loaded new snapshot from {ssd_storage_directory}, "
                f"HBM cache fully invalidated"
            )
