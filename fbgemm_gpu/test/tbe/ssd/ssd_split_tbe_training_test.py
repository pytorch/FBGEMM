# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3,6,56]

import tempfile
import unittest

from typing import Any, List, Optional

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BackendType,
    BoundsCheckMode,
    PoolingMode,
)

from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from hypothesis import assume, given, settings, Verbosity

from .. import common  # noqa E402
from ..common import gpu_unavailable, running_in_oss

from .training_common import (
    default_strategies,
    FlushLocation,
    MAX_EXAMPLES,
    MAX_PIPELINE_EXAMPLES,
    PrefetchLocation,
    SSDSplitTableBatchedEmbeddingsTestCommon,
)


KV_WORLD_SIZE = 4
VIRTUAL_TABLE_ROWS = int(
    2**18
)  # relatively large for now given optimizer is still pre-allocated


@unittest.skipIf(*running_in_oss)
@unittest.skipIf(*gpu_unavailable)
class SSDSplitTableBatchedEmbeddingsTest(SSDSplitTableBatchedEmbeddingsTestCommon):
    def get_physical_table_arg_indices_(self, feature_table_map: List[int]):
        """
        Get the physical table arg indices for the reference and TBE.  The
        first element in each tuple is for accessing the reference embedding
        list.  The second element is for accessing TBE data.

        Example:
            feature_table_map = [0, 1, 2, 2, 3, 4]
            This function returns [(0, 0), (1, 1), (2, 2), (4, 3), (5, 4)]
        """
        ref_arg_indices = []
        test_arg_indices = []
        prev_t = -1
        for f, t in enumerate(feature_table_map):
            # Only get the physical tables
            if prev_t != t:
                prev_t = t
                ref_arg_indices.append(f)
                test_arg_indices.append(t)
        return zip(ref_arg_indices, test_arg_indices)

    @given(
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        indice_int64_t=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd(self, indice_int64_t: bool, weights_precision: SparseType) -> None:
        import tempfile

        E = int(1e4)
        D = 128
        N = 100
        if indice_int64_t:
            indices = torch.as_tensor(
                np.random.choice(E, replace=False, size=(N,)), dtype=torch.int64
            )
        else:
            indices = torch.as_tensor(
                np.random.choice(E, replace=False, size=(N,)), dtype=torch.int32
            )
        count = torch.tensor([N])

        feature_table_map = list(range(1))
        emb = SSDTableBatchedEmbeddingBags(
            embedding_specs=[(E, D)],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=1,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            weights_precision=weights_precision,
            l2_cache_size=8,
        )

        weights = torch.randn(N, emb.cache_row_dim, dtype=weights_precision.as_dtype())
        output_weights = torch.empty_like(weights)

        emb.ssd_db.get_cuda(indices, output_weights, count)
        torch.cuda.synchronize()
        assert (output_weights <= 0.1).all().item()
        assert (output_weights >= -0.1).all().item()

        emb.ssd_db.set_cuda(indices, weights, count, 1)
        emb.ssd_db.get_cuda(indices, output_weights, count)
        torch.cuda.synchronize()
        torch.testing.assert_close(weights, output_weights)

    @given(
        **default_strategies,
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_forward(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        backend_type: BackendType,
    ) -> None:

        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # Generate embedding modules
        (
            emb,
            emb_ref,
        ) = self.generate_ssd_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            backend_type=backend_type,
        )

        # Generate inputs
        Es = [emb.embedding_specs[t][0] for t in range(T)]
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=trigger_bounds_check,
            mixed_B=mixed_B,
        )

        # Execute forward
        self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            weighted,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

    def execute_ssd_cache_pipeline_(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        prefetch_pipeline: bool,
        # If True, prefetch will be invoked by the user.
        explicit_prefetch: bool,
        prefetch_location: Optional[PrefetchLocation],
        use_prefetch_stream: bool,
        flush_location: Optional[FlushLocation],
        trigger_bounds_check: bool,
        mixed_B: bool = False,
        enable_raw_embedding_streaming: bool = False,
        num_iterations: int = 10,
    ) -> None:
        # If using pipeline prefetching, explicit prefetching must be True
        assert not prefetch_pipeline or explicit_prefetch

        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        torch.manual_seed(42)

        # Generate embedding modules
        (
            emb,
            emb_ref,
        ) = self.generate_ssd_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            # Disable stochastic rounding because error is too large when
            # running for many iterations. This should be OK for testing the
            # functionality of the cache
            stochastic_rounding=False,
            share_table=share_table,
            prefetch_pipeline=prefetch_pipeline,
            enable_raw_embedding_streaming=enable_raw_embedding_streaming,
        )

        optimizer_states_ref = [
            [s.clone().float() for s in states]
            for states in self.split_optimizer_states_(emb)
        ]

        Es = [emb.embedding_specs[t][0] for t in range(T)]

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        batches = []
        for _it in range(num_iterations):
            batches.append(
                self.generate_inputs_(
                    B,
                    L,
                    Es,
                    emb.feature_table_map,
                    weights_precision=weights_precision,
                    trigger_bounds_check=trigger_bounds_check,
                    mixed_B=mixed_B,
                )
            )

        prefetch_stream = (
            torch.cuda.Stream() if use_prefetch_stream else torch.cuda.current_stream()
        )
        forward_stream = torch.cuda.current_stream() if use_prefetch_stream else None

        force_flush = flush_location == FlushLocation.ALL

        if force_flush or flush_location == FlushLocation.BEFORE_TRAINING:
            emb.flush()

        # pyre-ignore[53]
        def _prefetch(b_it: int) -> int:
            if not explicit_prefetch or b_it >= num_iterations:
                return b_it + 1

            (
                _,
                _,
                indices,
                offsets,
                _,
                batch_size_per_feature_per_rank,
            ) = batches[b_it]
            with torch.cuda.stream(prefetch_stream):
                emb.prefetch(
                    indices,
                    offsets,
                    forward_stream=forward_stream,
                    batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
                )
            return b_it + 1

        if prefetch_pipeline:
            # Prefetch the first iteration
            _prefetch(0)
            b_it = 1
        else:
            b_it = 0

        for it in range(num_iterations):
            (
                indices_list,
                per_sample_weights_list,
                indices,
                offsets,
                per_sample_weights,
                batch_size_per_feature_per_rank,
            ) = batches[it]

            # Ensure that prefetch i is done before forward i
            if use_prefetch_stream:
                assert forward_stream is not None
                forward_stream.wait_stream(prefetch_stream)

            # Prefetch before forward
            if (
                not prefetch_pipeline
                or prefetch_location == PrefetchLocation.BEFORE_FWD
            ):
                b_it = _prefetch(b_it)

            # Execute forward
            output_ref_list, output = self.execute_ssd_forward_(
                emb,
                emb_ref,
                indices_list,
                per_sample_weights_list,
                indices,
                offsets,
                per_sample_weights,
                B,
                L,
                weighted,
                tolerance=tolerance,
                it=it,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )

            if force_flush or flush_location == FlushLocation.AFTER_FWD:
                emb.flush()

            # Prefetch between forward and backward
            if (
                prefetch_pipeline
                and prefetch_location == PrefetchLocation.BETWEEN_FWD_BWD
            ):
                b_it = _prefetch(b_it)

            # Zero out weight grad
            for f, _ in self.get_physical_table_arg_indices_(emb.feature_table_map):
                emb_ref[f].weight.grad = None

            # Execute backward
            self.execute_ssd_backward_(
                output_ref_list,
                output,
                B,
                D,
                pooling_mode,
                batch_size_per_feature_per_rank,
            )

            if force_flush or flush_location == FlushLocation.AFTER_BWD:
                emb.flush()

            # Compare optimizer states
            split_optimizer_states = self.split_optimizer_states_(emb)
            for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
                (optim_state_r,) = optimizer_states_ref[t]
                (optim_state_t,) = split_optimizer_states[t]
                emb_r = emb_ref[f]

                optim_state_r.add_(
                    # pyre-fixme[16]: `Optional` has no attribute `float`.
                    emb_r.weight.grad.float()
                    .to_dense()
                    .pow(2)
                    .mean(dim=1)
                )
                torch.testing.assert_close(
                    optim_state_t.float(),
                    optim_state_r,
                    atol=tolerance,
                    rtol=tolerance,
                )

                new_ref_weight = torch.addcdiv(
                    emb_r.weight.float(),
                    value=-lr,
                    tensor1=emb_r.weight.grad.float().to_dense(),
                    tensor2=optim_state_t.float().sqrt().add(eps).view(Es[t], 1),
                )

                if weights_precision == SparseType.FP16:
                    # Round the reference weight the same way that
                    # TBE does
                    new_ref_weight = new_ref_weight.half().float()
                    assert new_ref_weight.dtype == emb_r.weight.dtype

                emb_r.weight.data.copy_(new_ref_weight)

        # Compare weights
        emb.flush()
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            weight_r = emb_ref[f].weight.float()
            weight_t = emb.debug_split_embedding_weights()[t].float().cuda()
            torch.testing.assert_close(
                weight_t,
                weight_r,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        flush_location=st.sampled_from(FlushLocation),
        explicit_prefetch=st.booleans(),
        prefetch_location=st.sampled_from(PrefetchLocation),
        use_prefetch_stream=st.booleans(),
        **default_strategies,
    )
    def test_ssd_cache_flush(self, **kwargs: Any):
        """
        Test the correctness of the SSD cache prefetch workflow with
        excessive flushing
        """
        kwargs["prefetch_pipeline"] = False
        if kwargs["explicit_prefetch"] or not kwargs["use_prefetch_stream"]:
            kwargs["prefetch_pipeline"] = True

        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(kwargs["prefetch_pipeline"] and kwargs["explicit_prefetch"])
        assume(not kwargs["use_prefetch_stream"] or kwargs["prefetch_pipeline"])
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        self.execute_ssd_cache_pipeline_(
            **kwargs,
        )

    @given(**default_strategies)
    @settings(
        verbosity=Verbosity.verbose, max_examples=MAX_PIPELINE_EXAMPLES, deadline=None
    )
    def test_ssd_cache_implicit_prefetch(self, **kwargs: Any):
        """
        Test the correctness of the SSD cache prefetch workflow
        without pipeline prefetching and with implicit prefetching.
        Implicit prefetching relies on TBE forward to invoke prefetch.
        """
        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        self.execute_ssd_cache_pipeline_(
            prefetch_pipeline=False,
            explicit_prefetch=False,
            prefetch_location=None,
            use_prefetch_stream=False,
            flush_location=None,
            **kwargs,
        )

    @given(**default_strategies)
    @settings(
        verbosity=Verbosity.verbose, max_examples=MAX_PIPELINE_EXAMPLES, deadline=None
    )
    def test_ssd_cache_explicit_prefetch(self, **kwargs: Any):
        """
        Test the correctness of the SSD cache prefetch workflow
        without pipeline prefetching and with explicit prefetching
        (the user explicitly invokes prefetch).  Each prefetch invoked
        before a forward TBE fetches data for that specific iteration.

        For example:

        ------------------------- Timeline ------------------------>
        pf(i) -> fwd(i) -> ... -> pf(i+1) -> fwd(i+1) -> ...

        Note:
        - pf(i) = prefetch of iteration i
        - fwd(i) = forward TBE of iteration i
        """
        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        self.execute_ssd_cache_pipeline_(
            prefetch_pipeline=False,
            explicit_prefetch=True,
            prefetch_location=None,
            use_prefetch_stream=False,
            flush_location=None,
            **kwargs,
        )

    @given(use_prefetch_stream=st.booleans(), **default_strategies)
    @settings(
        verbosity=Verbosity.verbose, max_examples=MAX_PIPELINE_EXAMPLES, deadline=None
    )
    def test_ssd_cache_pipeline_before_fwd(self, **kwargs: Any):
        """
        Test the correctness of the SSD cache prefetch workflow with
        pipeline prefetching when cache prefetching of the next
        iteration is invoked before the forward TBE of the current
        iteration.

        For example:

        ------------------------- Timeline ------------------------>
        pf(i+1) -> fwd(i) -> ... -> pf(i+2) -> fwd(i+1) -> ...

        Note:
        - pf(i) = prefetch of iteration i
        - fwd(i) = forward TBE of iteration i
        """
        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        self.execute_ssd_cache_pipeline_(
            prefetch_pipeline=True,
            explicit_prefetch=True,
            prefetch_location=PrefetchLocation.BEFORE_FWD,
            flush_location=None,
            **kwargs,
        )

    @given(use_prefetch_stream=st.booleans(), **default_strategies)
    @settings(
        verbosity=Verbosity.verbose, max_examples=MAX_PIPELINE_EXAMPLES, deadline=None
    )
    def test_ssd_cache_pipeline_between_fwd_bwd(self, **kwargs: Any):
        """
        Test the correctness of the SSD cache prefetch workflow with
        pipeline prefetching when cache prefetching of the next
        iteration is invoked after the forward TBE and before the
        backward TBE of the current iteration.

        For example:

        ------------------------- Timeline ------------------------>
        fwd(i) -> pf(i+1) -> bwd(i) -> ... -> fwd(i+1) -> pf(i+2) -> bwd(i+1) -> ...

        Note:
        - pf(i) = prefetch of iteration i
        - fwd(i) = forward TBE of iteration i
        - bwd(i) = backward TBE of iteration i
        """

        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        self.execute_ssd_cache_pipeline_(
            prefetch_pipeline=True,
            explicit_prefetch=True,
            prefetch_location=PrefetchLocation.BETWEEN_FWD_BWD,
            flush_location=None,
            **kwargs,
        )

    @given(
        **default_strategies,
        num_buckets=st.integers(min_value=10, max_value=15),
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_db_forward(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        num_buckets: int,
        backend_type: BackendType,
    ) -> None:
        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)
        # Generate embedding modules
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            num_buckets=num_buckets,
            backend_type=backend_type,
        )

        # Generate inputs
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=trigger_bounds_check,
            mixed_B=mixed_B,
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
        )

        # Execute forward
        self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            weighted,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

    @given(
        **default_strategies,
        num_buckets=st.integers(min_value=10, max_value=15),
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
        enable_optimizer_offloading=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_apply_kv_state_dict(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        num_buckets: int,
        backend_type: BackendType,
        enable_optimizer_offloading: bool,
    ) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # TODO: check split_optimizer_states when optimizer offloading is ready
        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
            backend_type=backend_type,
        )

        # Generate inputs
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=trigger_bounds_check,
            mixed_B=mixed_B,
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
        )

        # Execute forward
        output_ref_list, output = self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            weighted,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict_list, bucket_asc_ids_list, num_active_id_per_bucket_list, _ = (
            emb.split_embedding_weights(no_snapshot=False, should_flush=True)
        )
        split_optimizer_states = emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False, should_flush=True
        )

        # create an empty emb with same parameters
        # Construct feature_table_map

        cache_sets = max(int(max(T * B * L, 1) * cache_set_scale), 1)
        emb2 = SSDTableBatchedEmbeddingBags(
            embedding_specs=emb.embedding_specs,
            feature_table_map=emb.feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=cache_sets,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            learning_rate=lr,
            eps=eps,
            ssd_rocksdb_shards=ssd_shards,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            stochastic_rounding=True,
            prefetch_pipeline=False,
            bounds_check_mode=BoundsCheckMode.WARNING,
            l2_cache_size=8,
            backend_type=backend_type,
            kv_zch_params=emb.kv_zch_params,
        ).cuda()

        # pyre-fixme[16]: Undefined attribute: `Optional` has no attribute `__iter__`.
        emb2.local_weight_counts = [ids.numel() for ids in bucket_asc_ids_list]
        emb2.enable_load_state_dict_mode()
        self.assertIsNotNone(emb2._cached_kvzch_data)
        for i, _ in enumerate(emb.embedding_specs):
            # pyre-ignore [16]
            emb2._cached_kvzch_data.cached_weight_tensor_per_table[i].copy_(
                # pyre-fixme[16]: Undefined attribute: Item `torch._tensor.Tensor` of `typing.Uni...
                emb_state_dict_list[i].full_tensor()
            )
            # NOTE: The [0] index is a hack since the test is fixed to use
            # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
            # be upgraded in the future to support multiple optimizers
            # pyre-ignore [16]
            emb2._cached_kvzch_data.cached_optimizer_states_per_table[i][0].copy_(
                split_optimizer_states[i][0]
            )
            # pyre-ignore [16]
            emb2._cached_kvzch_data.cached_id_tensor_per_table[i].copy_(
                # pyre-fixme[16]: Undefined attribute: `Optional` has no attribute `__getitem__`.
                bucket_asc_ids_list[i]
            )
            # pyre-ignore [16]
            emb2._cached_kvzch_data.cached_bucket_splits[i].copy_(
                num_active_id_per_bucket_list[i]
            )

        emb2.apply_state_dict()

        emb2.flush(True)
        # Compare emb state dict with expected values from nn.EmbeddingBag
        (
            emb_state_dict_list2,
            bucket_asc_ids_list2,
            num_active_id_per_bucket_list2,
            _,
        ) = emb2.split_embedding_weights(no_snapshot=False, should_flush=True)
        split_optimizer_states2 = emb2.split_optimizer_states(
            bucket_asc_ids_list2, no_snapshot=False, should_flush=True
        )

        for t in range(len(emb.embedding_specs)):
            sorted_ids = torch.sort(bucket_asc_ids_list[t].flatten())
            sorted_ids2 = torch.sort(bucket_asc_ids_list2[t].flatten())
            torch.testing.assert_close(
                sorted_ids.values,
                sorted_ids2.values,
                atol=tolerance,
                rtol=tolerance,
            )

            torch.testing.assert_close(
                # pyre-ignore [16]
                emb_state_dict_list[t].full_tensor()[sorted_ids.indices],
                # pyre-ignore [16]
                emb_state_dict_list2[t].full_tensor()[sorted_ids2.indices],
                atol=tolerance,
                rtol=tolerance,
            )
            torch.testing.assert_close(
                split_optimizer_states[t][0][sorted_ids.indices],
                split_optimizer_states2[t][0][sorted_ids2.indices],
                atol=tolerance,
                rtol=tolerance,
            )
            torch.testing.assert_close(
                num_active_id_per_bucket_list[t],
                num_active_id_per_bucket_list2[t],
                atol=tolerance,
                rtol=tolerance,
            )

    def _check_raw_embedding_stream_call_counts(
        self,
        mock_raw_embedding_stream: unittest.mock.Mock,
        mock_raw_embedding_stream_sync: unittest.mock.Mock,
        num_iterations: int,
        prefetch_pipeline: bool,
        L: int,
    ) -> None:
        offset = 2 if prefetch_pipeline else 1
        self.assertEqual(
            mock_raw_embedding_stream.call_count,
            num_iterations * 2 - offset if L > 0 else num_iterations - offset,
        )
        self.assertEqual(
            mock_raw_embedding_stream_sync.call_count, num_iterations - offset
        )

    @staticmethod
    def _record_event_mock(
        stream: torch.cuda.Stream,
        pre_event: Optional[torch.cuda.Event],
        post_event: Optional[torch.cuda.Event],
        **kwargs_: Any,
    ) -> None:
        with torch.cuda.stream(stream):
            if pre_event is not None:
                stream.wait_event(pre_event)

            if post_event is not None:
                stream.record_event(post_event)

    @given(
        use_prefetch_stream=st.booleans(),
        prefetch_location=st.sampled_from(PrefetchLocation),
        **default_strategies,
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_PIPELINE_EXAMPLES,
        deadline=None,
    )
    def test_raw_embedding_streaming(
        self,
        **kwargs: Any,
    ):
        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        num_iterations = 10
        prefetch_pipeline = False
        with unittest.mock.patch.object(
            SSDTableBatchedEmbeddingBags,
            "raw_embedding_stream",
            side_effect=self._record_event_mock,
        ) as mock_raw_embedding_stream, unittest.mock.patch.object(
            SSDTableBatchedEmbeddingBags,
            "raw_embedding_stream_sync",
            side_effect=self._record_event_mock,
        ) as mock_raw_embedding_stream_sync:
            self.execute_ssd_cache_pipeline_(
                prefetch_pipeline=prefetch_pipeline,
                explicit_prefetch=prefetch_pipeline,
                enable_raw_embedding_streaming=True,
                flush_location=None,
                num_iterations=num_iterations,
                **kwargs,
            )
            self._check_raw_embedding_stream_call_counts(
                mock_raw_embedding_stream=mock_raw_embedding_stream,
                mock_raw_embedding_stream_sync=mock_raw_embedding_stream_sync,
                num_iterations=num_iterations,
                prefetch_pipeline=prefetch_pipeline,
                L=kwargs["L"],
            )

    @given(
        use_prefetch_stream=st.booleans(),
        prefetch_location=st.sampled_from(PrefetchLocation),
        **default_strategies,
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_PIPELINE_EXAMPLES,
        deadline=None,
    )
    def test_raw_embedding_streaming_prefetch_pipeline(
        self,
        **kwargs: Any,
    ):
        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        num_iterations = 10
        prefetch_pipeline = True
        with unittest.mock.patch.object(
            SSDTableBatchedEmbeddingBags,
            "raw_embedding_stream",
            side_effect=self._record_event_mock,
        ) as mock_raw_embedding_stream, unittest.mock.patch.object(
            SSDTableBatchedEmbeddingBags,
            "raw_embedding_stream_sync",
            side_effect=self._record_event_mock,
        ) as mock_raw_embedding_stream_sync:
            self.execute_ssd_cache_pipeline_(
                prefetch_pipeline=prefetch_pipeline,
                explicit_prefetch=prefetch_pipeline,
                enable_raw_embedding_streaming=True,
                flush_location=None,
                num_iterations=num_iterations,
                **kwargs,
            )
            self._check_raw_embedding_stream_call_counts(
                mock_raw_embedding_stream=mock_raw_embedding_stream,
                mock_raw_embedding_stream_sync=mock_raw_embedding_stream_sync,
                num_iterations=num_iterations,
                prefetch_pipeline=prefetch_pipeline,
                L=kwargs["L"],
            )

    @given(
        **default_strategies,
        num_buckets=st.integers(min_value=10, max_value=15),
        backend_type=st.sampled_from([BackendType.DRAM]),
        enable_optimizer_offloading=st.booleans(),
        prefetch_pipeline=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_direct_write_embedding(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        num_buckets: int,
        backend_type: BackendType,
        enable_optimizer_offloading: bool,
        prefetch_pipeline: bool,
    ) -> None:
        """
        Test the direct_write_embedding function which writes weights directly to L1 cache,
        scratch pad, and backend without relying on auto-gradient.
        """

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            prefetch_pipeline=prefetch_pipeline,
            backend_type=backend_type,
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
            embedding_cache_mode=True,
        )

        # Generate inputs
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=trigger_bounds_check,
            mixed_B=mixed_B,
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
        )

        # Call prefetch explicitly if prefetch_pipeline is enabled
        if prefetch_pipeline:
            # For prefetch, we need to ensure the offsets match the expected format
            # The format should be B * T + 1 where B is batch size and T is number of tables
            # For mixed batch sizes, use the batch_size_per_feature_per_rank
            emb.prefetch(
                indices,
                offsets,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
            torch.cuda.synchronize()

        # Execute forward
        output_ref_list, output = self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            weighted,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

        # Create custom weights to write
        # First, create a mapping from linearized indices to weights
        # This ensures that the same index gets the same weight
        linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
            emb.hash_size_cumsum,
            indices,
            offsets,
            None,  # B_offsets
            -1,  # max_B
        )

        # Get unique indices
        unique_indices, inverse_indices = torch.unique(
            linear_cache_indices, return_inverse=True, sorted=True
        )

        # Create random weights for each unique index
        unique_weights = torch.randn(
            unique_indices.numel(),
            emb.cache_row_dim,
            device=emb.current_device,
            dtype=emb.weights_precision.as_dtype(),
        )

        # Map the unique weights back to the original indices
        custom_weights = unique_weights[inverse_indices]

        # Call direct_write_embedding
        emb.direct_write_embedding(
            indices=indices,
            offsets=offsets,
            weights=custom_weights,
        )
        torch.cuda.synchronize()

        # Verify weights were written correctly

        # 1. Check L1 cache
        # Get the cache locations for the indices
        linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
            emb.hash_size_cumsum,
            indices,
            offsets,
            None,  # B_offsets
            -1,  # max_B
        )

        lxu_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
            linear_cache_indices,
            emb.lxu_cache_state,
            emb.total_hash_size,
        )
        cache_location_mask = lxu_cache_locations >= 0

        # Get the cache locations and corresponding weights
        cache_locations = lxu_cache_locations[cache_location_mask]
        cache_weights = custom_weights[cache_location_mask]

        # Check that weights in L1 cache match the custom weights
        if cache_locations.numel() > 0:
            actual_cache_weights = emb.lxu_cache_weights[cache_locations]
            torch.testing.assert_close(
                actual_cache_weights,
                cache_weights,
                rtol=1e-2 if weights_precision == SparseType.FP16 else 1e-4,
                atol=1e-2 if weights_precision == SparseType.FP16 else 1e-4,
            )

        # 2. Check backend (SSD/DRAM)
        # Flush to ensure all weights are written to backend
        # emb.flush()

        # For indices not in L1 cache, verify they were written to backend
        # Directly get indices not in the cache
        non_cache_indices = linear_cache_indices[~cache_location_mask]
        non_cache_weights = custom_weights[~cache_location_mask]

        # Only proceed if there are indices not in the cache
        if non_cache_indices.numel() > 0:

            # Create a tensor to hold weights fetched from backend
            output_weights = torch.empty_like(non_cache_weights).cpu()

            # Fetch weights from backend using the same ssd_eviction_stream
            count = torch.tensor(
                [non_cache_indices.numel()], dtype=torch.int64, device="cpu"
            )

            # Use the ssd_eviction_stream for get_cuda to ensure proper synchronization with set_cuda
            with torch.cuda.stream(emb.ssd_eviction_stream):
                emb.ssd_db.get_cuda(non_cache_indices.cpu(), output_weights, count)

            # Synchronize to ensure the get_cuda operation is complete before comparing
            torch.cuda.synchronize()

            fetched_weights = output_weights.cuda()

            # Check that weights in backend match the custom weights
            torch.testing.assert_close(
                fetched_weights,
                non_cache_weights,
                rtol=1e-2 if weights_precision == SparseType.FP16 else 1e-4,
                atol=1e-2 if weights_precision == SparseType.FP16 else 1e-4,
            )

        # 3. Check scratch pad updates when prefetch_pipeline is enabled
        if prefetch_pipeline:
            # If prefetch_pipeline is enabled, direct_write_embedding should have:
            # 1. Called _update_cache_counter_and_pointers to run backward hooks for prefetch
            # 2. Popped the current scratch pad
            # 3. If there's a next batch scratch pad, written to it

            # Check if the scratch pad data is available in the SSD TBE implementation
            if (
                hasattr(emb, "ssd_scratch_pad_eviction_data")
                and len(emb.ssd_scratch_pad_eviction_data) > 0
            ):
                # Get the scratch pad data structure
                # The structure is a list of tuples, where each tuple contains:
                # - [0]: The sparse tensor (sp)
                # - [1]: The indices tensor (sp_idx)
                # - [2]: The actions count tensor
                sp_data = emb.ssd_scratch_pad_eviction_data[0]

                # Check if we have indices in the scratch pad
                if len(sp_data) >= 2 and sp_data[1] is not None:
                    # Get the indices in the scratch pad
                    sp_indices = sp_data[1].to(emb.current_device)

                    if sp_indices.numel() > 0:
                        # Create a set of linearized indices for easier comparison
                        linear_indices_set = set(
                            linear_cache_indices.detach().cpu().numpy().tolist()
                        )
                        sp_indices_set = set(sp_indices.detach().cpu().numpy().tolist())

                        # Check that all indices in the scratch pad are from our custom weights
                        # Note: The scratch pad might not contain all indices, as some might be in L1 cache
                        self.assertTrue(
                            sp_indices_set.issubset(linear_indices_set),
                            "Scratch pad indices should be a subset of the linearized indices",
                        )

                        # Check that the weights in the scratch pad match the custom weights
                        # Get the sparse tensor from the scratch pad data
                        sp = sp_data[0]

                        # For each index in the scratch pad, find its position in the linearized indices
                        # and check that the weight matches
                        for i, idx in enumerate(sp_indices):
                            idx_val = idx.item()
                            if idx_val in linear_indices_set:
                                # Find the position of this index in the linearized indices
                                pos = (linear_cache_indices == idx_val).nonzero(
                                    as_tuple=True
                                )[0][0]

                                # Get the weight from the scratch pad
                                sp_weight = sp[i]

                                # Get the corresponding custom weight
                                custom_weight = custom_weights[pos]

                                # Check that the weights match
                                torch.testing.assert_close(
                                    sp_weight,
                                    custom_weight,
                                    rtol=(
                                        1e-2
                                        if weights_precision == SparseType.FP16
                                        else 1e-4
                                    ),
                                    atol=(
                                        1e-2
                                        if weights_precision == SparseType.FP16
                                        else 1e-4
                                    ),
                                )
