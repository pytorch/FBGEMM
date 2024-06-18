#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[13,56]

import functools
import itertools
import logging
import os
import tempfile
from math import log2
from typing import List, Optional, Tuple, Type

import torch  # usort:skip

import fbgemm_gpu.split_embedding_codegen_lookup_invokers as invokers
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
    SplitState,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    apply_split_helper,
    CounterBasedRegularizationDefinition,
    CowClipDefinition,
    WeightDecayMode,
)

from torch import distributed as dist, nn, Tensor  # usort:skip
from torch.autograd.profiler import record_function

from .common import ASSOC


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
        ssd_block_cache_size: int = 0,
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
        # Parameter Server Configs
        ps_hosts: Optional[Tuple[Tuple[str, int]]] = None,
        tbe_unique_id: int = -1,
        ps_max_key_per_request: Optional[int] = None,
        ps_client_thread_num: Optional[int] = None,
        ps_max_local_index_length: Optional[int] = None,
    ) -> None:
        super(SSDTableBatchedEmbeddingBags, self).__init__()

        self.pooling_mode = pooling_mode
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

        assert cache_sets > 0
        element_size = weights_precision.bit_rate() // 8
        assert (
            element_size == 4 or element_size == 2
        ), f"Invalid element size {element_size}"
        cache_size = cache_sets * ASSOC * element_size * self.max_D
        logging.info(
            f"Using cache for SSD with admission algorithm "
            f"{CacheAlgorithm.LRU}, {cache_sets} sets, stored on {'DEVICE' if ssd_cache_location is EmbeddingLocation.DEVICE else 'MANAGED'} with {ssd_shards} shards, "
            f"SSD storage directory: {ssd_storage_directory}, "
            f"Memtable Flush Period: {ssd_memtable_flush_period}, "
            f"Memtable Flush Offset: {ssd_memtable_flush_offset}, "
            f"Desired L0 files per compaction: {ssd_l0_files_per_compact}, "
            f"{cache_size / 1024.0 / 1024.0 / 1024.0 : .2f}GB, "
            f"weights precision: {weights_precision}, "
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

        assert weights_precision in (SparseType.FP32, SparseType.FP16)
        assert output_dtype in (SparseType.FP32, SparseType.FP16)
        self.weights_precision = weights_precision
        self.output_dtype: int = output_dtype.as_int()

        cache_dtype = weights_precision.as_dtype()
        if ssd_cache_location == EmbeddingLocation.MANAGED:
            self.register_buffer(
                "lxu_cache_weights",
                torch.ops.fbgemm.new_managed_tensor(
                    torch.zeros(
                        1,
                        device=self.current_device,
                        dtype=cache_dtype,
                    ),
                    [cache_sets * ASSOC, self.max_D],
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

        self.timestep = 0

        os.makedirs(ssd_storage_directory, exist_ok=True)

        ssd_directory = tempfile.mkdtemp(
            prefix="ssd_table_batched_embeddings", dir=ssd_storage_directory
        )
        # logging.info("DEBUG: weights_precision {}".format(weights_precision))
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
                self.max_D,
                ssd_rate_limit_mbps,
                ssd_size_ratio,
                ssd_compaction_trigger,
                ssd_write_buffer_size,
                ssd_max_write_buffer_num,
                ssd_uniform_init_lower,
                ssd_uniform_init_upper,
                weights_precision.bit_rate(),  # row_storage_bitwidth
                ssd_block_cache_size,
            )
        else:
            # create tbe unique id using rank index | pooling mode
            if tbe_unique_id == -1:
                SSDTableBatchedEmbeddingBags._local_instance_index += 1
                assert (
                    SSDTableBatchedEmbeddingBags._local_instance_index < 8
                ), f"{SSDTableBatchedEmbeddingBags._local_instance_index}, more than 8 TBE instance is created in one rank, the tbe unique id won't be unique in this case."
                tbe_unique_id = (
                    dist.get_rank() << 3
                    | SSDTableBatchedEmbeddingBags._local_instance_index
                )
            logging.info(f"tbe_unique_id: {tbe_unique_id}")
            logging.info(f"ps_max_local_index_length: {ps_max_local_index_length}")
            logging.info(f"ps_client_thread_num: {ps_client_thread_num}")
            logging.info(f"ps_max_key_per_request: {ps_max_key_per_request}")
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
            )
        # pyre-fixme[20]: Argument `self` expected.
        (low_priority, high_priority) = torch.cuda.Stream.priority_range()
        self.ssd_stream = torch.cuda.Stream(priority=low_priority)
        self.ssd_set_start = torch.cuda.Event()
        self.ssd_set_end = torch.cuda.Event()
        self.timesteps_prefetched: List[int] = []
        self.ssd_scratch_pads: List[Tuple[Tensor, Tensor, Tensor]] = []
        # TODO: add type annotation
        self.ssd_prefetch_data = []

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
            dtype=torch.float32,
        )

        # add placeholder require_grad param to enable autograd without nn.parameter
        # this is needed to enable int8 embedding weights for SplitTableBatchedEmbedding
        self.placeholder_autograd_tensor = nn.Parameter(
            torch.zeros(0, device=self.current_device, dtype=torch.float)
        )

        # Register backward hook for evicting rows from a scratch pad to SSD
        # post backward
        self.placeholder_autograd_tensor.register_hook(self._evict_from_scratch_pad)

        assert optimizer in (
            OptimType.EXACT_ROWWISE_ADAGRAD,
        ), f"Optimizer {optimizer} is not supported by SSDTableBatchedEmbeddingBags"
        self.optimizer = optimizer

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

    def evict(
        self, evicted_rows: Tensor, evicted_indices: Tensor, actions_count_cpu: Tensor
    ) -> None:
        """
        Evict data from the given input tensors to SSD via RocksDB
        """
        with torch.cuda.stream(self.ssd_stream):
            self.ssd_stream.wait_event(self.ssd_set_start)
            evicted_rows_cpu = self.to_pinned_cpu(evicted_rows)
            evicted_indices_cpu = self.to_pinned_cpu(evicted_indices)
            evicted_rows.record_stream(self.ssd_stream)
            evicted_indices.record_stream(self.ssd_stream)
            self.ssd_db.set_cuda(
                evicted_indices_cpu, evicted_rows_cpu, actions_count_cpu, self.timestep
            )
            # TODO: is this needed?
            # Need a way to synchronize
            #  actions_count_cpu.record_stream(self.ssd_stream)
            self.ssd_stream.record_event(self.ssd_set_end)

    def _evict_from_scratch_pad(self, grad: Tensor) -> None:
        assert len(self.ssd_scratch_pads) > 0, "There must be at least one scratch pad"
        (inserted_rows_gpu, post_bwd_evicted_indices, actions_count_cpu) = (
            self.ssd_scratch_pads.pop(0)
        )
        self.evict(inserted_rows_gpu, post_bwd_evicted_indices, actions_count_cpu)

    def _compute_cache_ptrs(
        self,
        linear_cache_indices: torch.Tensor,
        assigned_cache_slots: torch.Tensor,
        linear_index_inverse_indices: torch.Tensor,
        unique_indices_count_cumsum: torch.Tensor,
        cache_set_inverse_indices: torch.Tensor,
        inserted_rows_gpu: torch.Tensor,
        unique_indices_length: torch.Tensor,
        inserted_indices: torch.Tensor,
        actions_count_cpu: torch.Tensor,
    ) -> torch.Tensor:
        with record_function("## ssd_tbe_lxu_cache_lookup ##"):
            lxu_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
                linear_cache_indices,
                self.lxu_cache_state,
                self.total_hash_size,
            )

        with record_function("## ssd_generate_row_addrs ##"):
            lxu_cache_ptrs, post_bwd_evicted_indices = (
                torch.ops.fbgemm.ssd_generate_row_addrs(
                    lxu_cache_locations,
                    assigned_cache_slots,
                    linear_index_inverse_indices,
                    unique_indices_count_cumsum,
                    cache_set_inverse_indices,
                    self.lxu_cache_weights,
                    inserted_rows_gpu,
                    unique_indices_length,
                    inserted_indices,
                )
            )

        with record_function("## ssd_scratch_pads ##"):
            # Store scratch pad info for post backward eviction
            self.ssd_scratch_pads.append(
                (inserted_rows_gpu, post_bwd_evicted_indices, actions_count_cpu)
            )

        return (
            lxu_cache_ptrs,
            inserted_rows_gpu,
            post_bwd_evicted_indices,
            actions_count_cpu,
        )

    def prefetch(self, indices: Tensor, offsets: Tensor) -> Optional[Tensor]:
        (indices, offsets) = indices.long(), offsets.long()

        linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
            self.hash_size_cumsum,
            indices,
            offsets,
        )
        self.timestep += 1
        self.timesteps_prefetched.append(self.timestep)
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
        )

        actions_count_cpu = self.to_pinned_cpu(actions_count_gpu)
        assigned_cache_slots = assigned_cache_slots.long()
        evicted_rows = self.lxu_cache_weights[
            assigned_cache_slots.clamp(min=0).long(), :
        ]
        inserted_rows = torch.empty(
            evicted_rows.shape,
            dtype=self.lxu_cache_weights.dtype,
            pin_memory=True,
        )

        current_stream = torch.cuda.current_stream()

        # Ensure the previous iterations l3_db.set(..) has completed.
        current_stream.wait_event(self.ssd_set_end)
        self.ssd_db.get_cuda(
            self.to_pinned_cpu(inserted_indices), inserted_rows, actions_count_cpu
        )
        current_stream.record_event(self.ssd_set_start)
        # TODO: T123943415 T123943414 this is a big copy that is (mostly) unnecessary with a decent cache hit rate.
        # Should we allocate on HBM?
        inserted_rows_gpu = inserted_rows.cuda(non_blocking=True)

        torch.ops.fbgemm.masked_index_put(
            self.lxu_cache_weights,
            assigned_cache_slots,
            inserted_rows_gpu,
            actions_count_gpu,
        )

        # Evict rows from cache to SSD
        self.evict(evicted_rows, evicted_indices, actions_count_cpu)

        # TODO: keep only necessary tensors
        self.ssd_prefetch_data.append(
            (
                linear_cache_indices,
                assigned_cache_slots,
                linear_index_inverse_indices,
                unique_indices_count_cumsum,
                cache_set_inverse_indices,
                inserted_rows_gpu,
                unique_indices_length,
                inserted_indices,
                actions_count_cpu,
            )
        )

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
    ) -> Tensor:
        (indices, offsets) = indices.long(), offsets.long()
        # Force casting per_sample_weights to float
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.float()
        if len(self.timesteps_prefetched) == 0:
            with record_function("## prefetch ##"):
                self.prefetch(indices, offsets)
        assert len(self.ssd_prefetch_data) > 0

        prefetch_data = self.ssd_prefetch_data.pop(0)
        (
            lxu_cache_ptrs,
            inserted_rows_gpu,
            post_bwd_evicted_indices,
            actions_count_cpu,
        ) = self._compute_cache_ptrs(*prefetch_data)

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
            vbe_metadata=invokers.lookup_args_ssd.VBEMetadata(
                B_offsets=None,
                output_offsets_feature_rank=None,
                B_offsets_rank_per_feature=None,
                max_B=-1,
                max_B_feature_rank=-1,
                output_size=-1,
            ),
            # Unused arguments
            is_experimental=False,
            use_uniq_cache_locations_bwd=False,
            use_homogeneous_placements=True,
            # The keys for ssd_tensors are controlled by ssd_tensors in
            # codegen/genscript/optimizer_args.py
            ssd_tensors={
                "row_addrs": lxu_cache_ptrs,
                "inserted_rows": inserted_rows_gpu,
                "post_bwd_evicted_indices": post_bwd_evicted_indices,
                "actions_count": actions_count_cpu,
            },
        )

        self.timesteps_prefetched.pop(0)

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
    def set_learning_rate(self, lr: float) -> None:
        """
        Sets the learning rate.
        """
        self._set_learning_rate(lr)

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

        active_slots_mask_cpu = active_slots_mask.cpu()
        lxu_cache_weights_cpu = self.lxu_cache_weights.cpu()
        lxu_cache_state_cpu = self.lxu_cache_state.cpu()

        active_weights = lxu_cache_weights_cpu.masked_select(
            active_slots_mask_cpu.view(-1, 1)
        ).view(-1, self.max_D)
        active_ids = lxu_cache_state_cpu.view(-1).masked_select(
            active_slots_mask_cpu.view(-1)
        )

        torch.cuda.current_stream().wait_stream(self.ssd_stream)

        self.ssd_db.set_cuda(
            active_ids,
            active_weights,
            torch.tensor([active_ids.numel()]),
            self.timestep,
        )
