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
from typing import Any, Callable, List, Optional, Tuple, Type

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
        # Parameter Server Configs
        ps_hosts: Optional[Tuple[Tuple[str, int]]] = None,
        tbe_unique_id: int = -1,
        # in local test we need to use the pass in path for rocksdb creation
        # in production we need to do it inside SSD mount path which will ignores the passed in path
        use_passed_in_path: int = True,
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
            f"{CacheAlgorithm.LRU}, {cache_sets} sets, stored on {'DEVICE' if ssd_cache_location is EmbeddingLocation.DEVICE else 'MANAGED'} with {ssd_rocksdb_shards} shards, "
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
        if not ps_hosts:
            logging.info(
                f"Logging SSD offloading setup "
                f"passed_in_path={ssd_directory}, num_shards={ssd_rocksdb_shards},num_threads={ssd_rocksdb_shards},"
                f"memtable_flush_period={ssd_memtable_flush_period},memtable_flush_offset={ssd_memtable_flush_offset},"
                f"l0_files_per_compact={ssd_l0_files_per_compact},max_D={self.max_D},rate_limit_mbps={ssd_rate_limit_mbps},"
                f"size_ratio={ssd_size_ratio},compaction_trigger={ssd_compaction_trigger},"
                f"write_buffer_size_per_tbe={ssd_rocksdb_write_buffer_size},max_write_buffer_num_per_db_shard={ssd_max_write_buffer_num},"
                f"uniform_init_lower={ssd_uniform_init_lower},uniform_init_upper={ssd_uniform_init_upper},"
                f"row_storage_bitwidth={weights_precision.bit_rate()},block_cache_size_per_tbe={ssd_block_cache_size_per_tbe},"
                f"use_passed_in_path:{use_passed_in_path}, real_path will be printed in EmbeddingRocksDB"
            )
            # pyre-fixme[4]: Attribute must be annotated.
            # pyre-ignore[16]
            self.ssd_db = torch.classes.fbgemm.EmbeddingRocksDBWrapper(
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
            )
        else:
            # create tbe unique id using rank index | local tbe idx
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
            # pyre-fixme[4]: Attribute must be annotated.
            # pyre-ignore[16]
            self.ssd_db = torch.classes.fbgemm.EmbeddingParameterServerWrapper(
                [host[0] for host in ps_hosts],
                [host[1] for host in ps_hosts],
                tbe_unique_id,
                54,
                32,
            )
        # pyre-fixme[20]: Argument `self` expected.
        (low_priority, high_priority) = torch.cuda.Stream.priority_range()
        # GPU stream for SSD cache eviction
        self.ssd_eviction_stream = torch.cuda.Stream(priority=low_priority)
        # GPU stream for SSD memory copy
        self.ssd_memcpy_stream = torch.cuda.Stream(priority=low_priority)

        # SSD get completion event
        self.ssd_event_get = torch.cuda.Event()
        # SSD eviction completion event
        self.ssd_event_evict = torch.cuda.Event()
        # SSD backward completion event
        self.ssd_event_backward = torch.cuda.Event()
        # SSD scratch pad eviction completion event
        self.ssd_event_evict_sp = torch.cuda.Event()
        # SSD get's input copy completion event
        self.ssd_event_get_inputs_cpy = torch.cuda.Event()

        self.timesteps_prefetched: List[int] = []
        self.ssd_scratch_pads: List[Tuple[Tensor, Tensor, Tensor, bool]] = []
        # TODO: add type annotation
        # pyre-fixme[4]: Attribute must be annotated.
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

    def to_pinned_cpu_on_stream_wait_on_current_stream(
        self,
        tensors: List[Tensor],
        stream: torch.cuda.Stream,
        post_event: Optional[torch.cuda.Event] = None,
    ) -> List[Tensor]:
        """
        Transfer input tensors from GPU to CPU using a pinned host buffer.
        The transfer is carried out on the given stream (`stream`) after all
        the kernels in the default stream (`current_stream`) are complete.

        Args:
            tensors (List[Tensor]): The list of tensors to be transferred
            stream (Stream): The stream to run memory copy
            post_event (Event): The post completion event

        Returns:
            The list of pinned CPU tensors
        """
        current_stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            stream.wait_stream(current_stream)
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
        pre_event: torch.cuda.Event,
        post_event: torch.cuda.Event,
        is_rows_uvm: bool,
        name: Optional[str] = "",
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
                )

                # TODO: is this needed?
                # Need a way to synchronize
                #  actions_count_cpu.record_stream(self.ssd_eviction_stream)
                stream.record_event(post_event)

    def _evict_from_scratch_pad(self, grad: Tensor) -> None:
        assert len(self.ssd_scratch_pads) > 0, "There must be at least one scratch pad"
        (inserted_rows, post_bwd_evicted_indices_cpu, actions_count_cpu, do_evict) = (
            self.ssd_scratch_pads.pop(0)
        )
        if do_evict:
            torch.cuda.current_stream().record_event(self.ssd_event_backward)
            self.evict(
                rows=inserted_rows,
                indices_cpu=post_bwd_evicted_indices_cpu,
                actions_count_cpu=actions_count_cpu,
                stream=self.ssd_eviction_stream,
                pre_event=self.ssd_event_backward,
                post_event=self.ssd_event_evict_sp,
                is_rows_uvm=True,
                name="scratch_pad",
            )

    def _compute_cache_ptrs(
        self,
        linear_cache_indices: torch.Tensor,
        assigned_cache_slots: torch.Tensor,
        linear_index_inverse_indices: torch.Tensor,
        unique_indices_count_cumsum: torch.Tensor,
        cache_set_inverse_indices: torch.Tensor,
        inserted_rows: torch.Tensor,
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
                    inserted_rows,
                    unique_indices_length,
                    inserted_indices,
                )
            )

        # Transfer post_bwd_evicted_indices from GPU to CPU right away to
        # increase a chance of overlapping with compute in the default stream
        (post_bwd_evicted_indices_cpu,) = (
            self.to_pinned_cpu_on_stream_wait_on_current_stream(
                tensors=[post_bwd_evicted_indices],
                stream=self.ssd_eviction_stream,
                post_event=None,
            )
        )

        with record_function("## ssd_scratch_pads ##"):
            # Store scratch pad info for post backward eviction
            self.ssd_scratch_pads.append(
                (
                    inserted_rows,
                    post_bwd_evicted_indices_cpu,
                    actions_count_cpu,
                    linear_cache_indices.numel() > 0,
                )
            )

        # pyre-fixme[7]: Expected `Tensor` but got `Tuple[typing.Any, Tensor,
        #  typing.Any, Tensor]`.
        return (
            lxu_cache_ptrs,
            inserted_rows,
            post_bwd_evicted_indices_cpu,
            actions_count_cpu,
        )

    def prefetch(self, indices: Tensor, offsets: Tensor) -> Optional[Tensor]:
        with record_function("## ssd_prefetch ##"):
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
            # Transfer evicted indices from GPU to CPU right away to increase a
            # chance of overlapping with compute on the default stream
            (evicted_indices_cpu,) = (
                self.to_pinned_cpu_on_stream_wait_on_current_stream(
                    tensors=[evicted_indices],
                    stream=self.ssd_eviction_stream,
                    post_event=None,
                )
            )

            actions_count_cpu, inserted_indices_cpu = (
                self.to_pinned_cpu_on_stream_wait_on_current_stream(
                    tensors=[
                        actions_count_gpu,
                        inserted_indices,
                    ],
                    stream=self.ssd_memcpy_stream,
                    post_event=self.ssd_event_get_inputs_cpy,
                )
            )

            assigned_cache_slots = assigned_cache_slots.long()
            evicted_rows = self.lxu_cache_weights[
                assigned_cache_slots.clamp(min=0).long(), :
            ]

            if linear_cache_indices.numel() > 0:
                inserted_rows = torch.ops.fbgemm.new_managed_tensor(
                    torch.zeros(
                        1,
                        device=self.current_device,
                        dtype=self.lxu_cache_weights.dtype,
                    ),
                    evicted_rows.shape,
                )
            else:
                inserted_rows = torch.empty(
                    evicted_rows.shape,
                    dtype=self.lxu_cache_weights.dtype,
                    device=self.current_device,
                )

            current_stream = torch.cuda.current_stream()

            inserted_indices_cpu = self.to_pinned_cpu(inserted_indices)

            # Ensure the previous iterations l3_db.set(..) has completed.
            current_stream.wait_event(self.ssd_event_evict)
            current_stream.wait_event(self.ssd_event_evict_sp)
            current_stream.wait_event(self.ssd_event_get_inputs_cpy)

            if linear_cache_indices.numel() > 0:
                self.record_function_via_dummy_profile(
                    "## ssd_get ##",
                    self.ssd_db.get_cuda,
                    inserted_indices_cpu,
                    inserted_rows,
                    actions_count_cpu,
                )
            current_stream.record_event(self.ssd_event_get)

            torch.ops.fbgemm.masked_index_put(
                self.lxu_cache_weights,
                assigned_cache_slots,
                inserted_rows,
                actions_count_gpu,
            )

            if linear_cache_indices.numel() > 0:
                # Evict rows from cache to SSD
                self.evict(
                    rows=evicted_rows,
                    indices_cpu=evicted_indices_cpu,
                    actions_count_cpu=actions_count_cpu,
                    stream=self.ssd_eviction_stream,
                    pre_event=self.ssd_event_get,
                    post_event=self.ssd_event_evict,
                    is_rows_uvm=False,
                    name="cache",
                )

            # TODO: keep only necessary tensors
            self.ssd_prefetch_data.append(
                (
                    linear_cache_indices,
                    assigned_cache_slots,
                    linear_index_inverse_indices,
                    unique_indices_count_cumsum,
                    cache_set_inverse_indices,
                    inserted_rows,
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
            self.prefetch(indices, offsets)
        assert len(self.ssd_prefetch_data) > 0

        prefetch_data = self.ssd_prefetch_data.pop(0)
        (
            lxu_cache_ptrs,
            inserted_rows,
            post_bwd_evicted_indices_cpu,
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
                "inserted_rows": inserted_rows,
                "post_bwd_evicted_indices": post_bwd_evicted_indices_cpu,
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

        torch.cuda.current_stream().wait_stream(self.ssd_eviction_stream)

        self.ssd_db.set_cuda(
            active_ids,
            active_weights,
            torch.tensor([active_ids.numel()]),
            self.timestep,
        )
