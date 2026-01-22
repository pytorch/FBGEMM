#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import random
import unittest
from typing import Any, cast, Optional

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
    RecordCacheMetrics,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    MultiPassPrefetchConfig,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.utils import (
    generate_requests,
    get_table_batched_offsets_from_dense,
    TBERequest,
    to_device,
)
from hypothesis import assume, given, settings

from ..common import MAX_EXAMPLES  # noqa E402
from .cache_common import (
    assert_cache,
    generate_cache_tbes,
    gpu_unavailable,
    optests,
    running_on_github,
    running_on_rocm,
    TestingStatsReporter,
    TestingStatsReporterConfig,
    VERBOSITY,
)


@optests.generate_opcheck_tests(fast=True)
class CacheTest(unittest.TestCase):
    def _compute_grad_output_shape(
        self,
        B: int,
        D_offsets: list[int],
        mixed_B: bool,
        Bs_feature_rank: Optional[list[list[int]]] = None,
    ) -> tuple[int, ...]:
        """
        Compute output gradient shape
        If mixed_B = True (variable batch size), the shape is sum(Bi * Di for
            all i's), where Bi is the batch size of feature i and Di is the
            embedding dimension of feature i.
        Otherwise, the shape is (B, sum(Di for all i's)), where Di is the
            embedding dimension of feature i
        """
        if mixed_B:
            assert Bs_feature_rank is not None, "Bs_feature_rank must not be None"
            Bs = [sum(Bs_feature) for Bs_feature in Bs_feature_rank]
            Ds = [D_offsets[i + 1] - D_offsets[i] for i in range(len(D_offsets) - 1)]
            return (sum([B * D for B, D in zip(Bs, Ds)]),)
        else:
            return (B, D_offsets[-1])

    @optests.dontGenerateOpCheckTests("Serial OOM")
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_github)
    @unittest.skipIf(*running_on_rocm)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        mixed=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        weights_cache_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        stochastic_rounding=st.booleans(),
        gather_uvm_cache_stats=st.booleans(),
        mixed_B=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_pipeline(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
        cache_algorithm: CacheAlgorithm,
        weights_cache_precision: SparseType,
        stochastic_rounding: bool,
        gather_uvm_cache_stats: bool,
        mixed_B: bool,
    ) -> None:
        assume(weights_cache_precision == SparseType.FP16 or not stochastic_rounding)
        # Need more than one table for variable batch sizes
        assume(not mixed_B or T > 1)
        cc, cc_ref, min_Es, sum_Ds = generate_cache_tbes(
            T,
            D,
            log_E,
            mixed,
            cache_algorithm,
            weights_cache_precision=weights_cache_precision,
            stochastic_rounding=stochastic_rounding,
            gather_uvm_cache_stats=gather_uvm_cache_stats,
        )
        iters = 3
        vbe_num_ranks = random.randint(2, 5)
        requests = generate_requests(
            iters,
            max(B // vbe_num_ranks, 1) if mixed_B else B,
            T,
            L,
            min_Es,
            reuse=0.1,
            sigma_B=1 if mixed_B else None,
            vbe_num_ranks=vbe_num_ranks if mixed_B else None,
        )

        # Generate grad_output
        assert len(requests) > 0, "There must be at least one request"
        output_shape = self._compute_grad_output_shape(
            B,
            cc.D_offsets.detach().cpu().tolist(),
            mixed_B,
            requests[0].Bs_per_feature_per_rank,
        )
        grad_output = torch.randn(*output_shape).cuda()

        for req in requests:
            indices, offsets, _, Bs_feature_rank = req.unpack_4()
            output = cc(
                indices, offsets, batch_size_per_feature_per_rank=Bs_feature_rank
            )
            output_ref = cc_ref(
                indices, offsets, batch_size_per_feature_per_rank=Bs_feature_rank
            )
            assert_cache(output, output_ref, stochastic_rounding)
            output.backward(grad_output)
            output_ref.backward(grad_output)
        cc.flush()
        for t in range(T):
            assert_cache(
                cc.split_embedding_weights()[t],
                cc_ref.split_embedding_weights()[t],
                stochastic_rounding,
            )

    def _test_cache_prefetch_pipeline(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
        prefetch_location: str,
        prefetch_stream: Optional[torch.cuda.Stream],
        weights_cache_precision: SparseType,
        stochastic_rounding: bool,
        gather_uvm_cache_stats: bool,
        mixed_B: bool = False,
        mpp_n_passes: Optional[int] = None,
        mpp_min_size: Optional[int] = None,
        trigger_bounds_check: bool = False,
    ) -> None:
        """
        test cache prefetch pipeline with prefetch_pipeline=True.
        prefetch_location can be "before_fwd" or "between_fwd_bwd",
        where the TBE prefetch(batch_{i+1}) is called before forward(batch_i)
        or in between of forward(batch_i) and backward(batch_i), respectively.
        If prefetch_stream is not None, the TBE prefetch function will use this stream.
        In addition, we make the TBE weights initialized as integer values, learning_rate
        as integer value, and gradients as integer values so that the test is more stable.
        """
        if mixed_B and T == 1:
            return
        # Need more than one table for variable batch sizes
        assert prefetch_location in ["before_fwd", "between_fwd_bwd"]
        reporter = TestingStatsReporterConfig(interval=2)

        mpp_conf: Optional[MultiPassPrefetchConfig] = None
        if mpp_n_passes or mpp_min_size:
            mpp_conf = MultiPassPrefetchConfig()
            if mpp_n_passes:
                mpp_conf = mpp_conf._replace(num_passes=mpp_n_passes)
            if mpp_min_size:
                mpp_conf = mpp_conf._replace(min_splitable_pass_size=mpp_min_size)
        cc: SplitTableBatchedEmbeddingBagsCodegen
        cc, cc_ref, min_Es, sum_Ds = generate_cache_tbes(
            T,
            D,
            log_E,
            mixed,
            CacheAlgorithm.LRU,
            prefetch_pipeline=True,
            use_int_weight=True,
            weights_cache_precision=weights_cache_precision,
            stochastic_rounding=stochastic_rounding,
            gather_uvm_cache_stats=gather_uvm_cache_stats,
            reporter_config=reporter,
            multipass_prefetch_config=mpp_conf,
        )
        iters = 5
        vbe_num_ranks = random.randint(2, 5)
        requests = generate_requests(
            iters,
            max(B // vbe_num_ranks, 1) if mixed_B else B,
            T,
            L,
            min_Es,
            reuse=0.1,
            sigma_B=1 if mixed_B else None,
            vbe_num_ranks=vbe_num_ranks if mixed_B else None,
        )

        # Cast indices and offsets to long
        for i, req in enumerate(requests):
            indices, offsets, weights, Bs_feature_rank = req.unpack_4()
            requests[i] = TBERequest(
                indices.long(), offsets.long(), weights, Bs_feature_rank
            )

        if trigger_bounds_check:
            # Randomly set some indices to be out of bound
            for i, req in enumerate(requests):
                indices, offsets, weights, Bs_feature_rank = req.unpack_4()
                num_indices = indices.numel()
                pos = random.sample(range(0, num_indices), (num_indices + 9) // 10)
                indices[pos] = min_Es * 2
                requests[i] = TBERequest(indices, offsets, weights, Bs_feature_rank)

        # Generat grad_output
        assert len(requests) > 0, "There must be at least one request"
        output_shape = self._compute_grad_output_shape(
            B,
            cc.D_offsets.detach().cpu().tolist(),
            mixed_B,
            requests[0].Bs_per_feature_per_rank,
        )

        grad_output = (
            torch.randint(
                low=-10,
                high=10,
                size=output_shape,
            )
            .float()
            .cuda()
        )
        torch.cuda.synchronize()  # make sure TBEs and inputs are ready
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Union[bool,
        #  Tensor]`.
        self.assertTrue(torch.all(cc.lxu_cache_locking_counter == 0))

        cur_stream: torch.cuda.Stream = torch.cuda.current_stream()

        req_iter = iter(requests)
        batch_i = next(req_iter)
        batch_ip1 = None
        output, output_ref = None, None

        def _prefetch(
            cc: SplitTableBatchedEmbeddingBagsCodegen,
            batch: Optional[TBERequest],
        ) -> None:
            if not batch:
                return
            stream = cur_stream if prefetch_stream else None
            indices, offsets, _, Bs_feature_rank = batch.unpack_4()
            context_stream = prefetch_stream if prefetch_stream else cur_stream
            with torch.cuda.stream(context_stream):
                cc.prefetch(
                    indices,
                    offsets,
                    forward_stream=stream,
                    batch_size_per_feature_per_rank=Bs_feature_rank,
                )

        _prefetch(cc, batch_i)

        input_batch_count: list[int] = []
        intput_original_size: int = 0
        intput_long_size: int = 0
        output_batch_count: list[int] = []
        output_original_size: int = 0
        while batch_i:
            indices, offsets, _, Bs_feature_rank = batch_i.unpack_4()
            # We force the conversion because this is what TBE kernel did in forward
            intput_original_size = indices.element_size()
            intput_long_size = indices.long().element_size()
            input_batch_count.append(indices.numel())
            batch_ip1 = next(req_iter, None)
            if prefetch_stream:
                cur_stream.wait_stream(prefetch_stream)
            if prefetch_location == "before_fwd":
                _prefetch(cc, batch_ip1)
            output = cc(
                indices, offsets, batch_size_per_feature_per_rank=Bs_feature_rank
            )
            output_batch_count.append(output.numel())
            output_original_size = output.element_size()
            if prefetch_location == "between_fwd_bwd":
                _prefetch(cc, batch_ip1)
            output.backward(grad_output)
            batch_i = batch_ip1
            batch_ip1 = None
        cc.flush()

        for req in requests:
            indices, offsets, _, Bs_feature_rank = req.unpack_4()
            output_ref = cc_ref(
                indices, offsets, batch_size_per_feature_per_rank=Bs_feature_rank
            )
            output_ref.backward(grad_output)

        for t in range(T):
            assert_cache(
                cc.split_embedding_weights()[t],
                cc_ref.split_embedding_weights()[t],
                stochastic_rounding,
            )

        assert_cache(output, output_ref, stochastic_rounding)
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Union[bool,
        #  Tensor]`.
        self.assertTrue(torch.all(cc.lxu_cache_locking_counter == 0))

        if prefetch_stream:
            # We record timing info at batch 1, 3, 5

            # But before, we need to wait until all backwards finished. Then
            # force report for async timer
            torch.cuda.synchronize()
            assert cc.bwd_wait_prefetch_timer, "Timer must have been set"
            cc.bwd_wait_prefetch_timer._lazy_report()
            assert cc.prefetch_duration_timer, "Timer must have been set"
            cc.prefetch_duration_timer._lazy_report()

            self.assertIsInstance(cc.stats_reporter, TestingStatsReporter)
            stats_reporter: TestingStatsReporter = cast(
                TestingStatsReporter, cc.stats_reporter
            )

            def assert_event_exist(
                event_name: str,
                steps: list[int],
                expected_value: Optional[list[int]] = None,
            ) -> None:
                self.assertEqual(
                    len(stats_reporter.reported_data[event_name]), len(steps)
                )
                if expected_value:
                    self.assertEqual(len(expected_value), len(steps))
                for i, step in enumerate(steps):
                    (
                        rep_step,
                        rep_event,
                        rep_val,
                        rep_emb_id,
                        rep_tbe_id,
                    ) = stats_reporter.reported_data[event_name].pop(0)
                    self.assertEqual(rep_step, step)
                    self.assertEqual(rep_event, event_name)
                    if expected_value:
                        self.assertEqual(rep_val, expected_value[i])
                    else:
                        self.assertGreaterEqual(float(rep_val), 0)
                    self.assertEqual(rep_emb_id, cc.logging_table_name)
                    self.assertEqual(rep_tbe_id, cc.uuid)

            def assert_event_not_exist(event_name: str) -> None:
                self.assertFalse(event_name in stats_reporter.reported_data)

            # Any reporting event happen before forward() will bear step timestamp
            # of 1 ~ 5, only odd step will be reported, so 1, 3, 5 steps will be in
            #
            # On the other side, if a reporting event happens after forward(), it'll
            # have step timestamp 0 ~ 4, so only 1, 3 steps will be in.
            assert_event_exist("bwd_wait_for_prefetch", [1, 3, 5], [])
            assert_event_exist("total_prefetch_duration", [1, 3], [])
            # commented out to not break AMD CI
            # assert_event_exist(
            #     "tbe.total_hbm_usage",
            #     [1, 3, 5],
            # )
            # assert_event_exist(
            #     "tbe.total_uvm_usage",
            #     [1, 3, 5],
            # )
            assert_event_exist(
                "tbe.fwd_input_size",
                [1, 3, 5],
                [input_batch_count[i] * intput_long_size for i in [0, 2, 4]],
            )
            assert_event_exist(
                "tbe.fwd_input_count",
                [1, 3, 5],
                [input_batch_count[i] for i in [0, 2, 4]],
            )
            assert_event_exist(
                "tbe.fwd_output_size",
                [1, 3, 5],
                [output_batch_count[i] * output_original_size for i in [0, 2, 4]],
            )
            assert_event_exist(
                "tbe.fwd_output_count",
                [1, 3, 5],
                [output_batch_count[i] for i in [0, 2, 4]],
            )

            uvm_cache_events = [
                "tbe.prefetch.cache_stats_by_data_size.num_calls",
                "tbe.prefetch.cache_stats_by_data_size.num_requested_indices",
                "tbe.prefetch.cache_stats_by_data_size.num_unique_indices",
                "tbe.prefetch.cache_stats_by_data_size.num_unique_misses",
                "tbe.prefetch.cache_stats_by_data_size.num_conflict_unique_misses",
                "tbe.prefetch.cache_stats_by_data_size.num_conflict_misses",
            ]
            for event in uvm_cache_events:
                if gather_uvm_cache_stats:
                    assert_event_exist(event, [1, 3], [])
                else:
                    assert_event_not_exist(event)
            assert_event_exist(
                "tbe.prefetch_input_size",
                [1, 3],
                [input_batch_count[i] * intput_original_size for i in [1, 3]],
            )
            assert_event_exist(
                "tbe.prefetch_input_count",
                [1, 3],
                [input_batch_count[i] for i in [1, 3]],
            )

    @optests.dontGenerateOpCheckTests("Serial OOM")
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_github)
    @unittest.skipIf(*running_on_rocm)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        mixed=st.booleans(),
        prefetch_location=st.sampled_from(["before_fwd", "between_fwd_bwd"]),
        weights_cache_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        stochastic_rounding=st.booleans(),
        gather_uvm_cache_stats=st.booleans(),
        mpp_n_passes=st.sampled_from([None, 1, 6, 12]),
        mpp_min_size=st.sampled_from([None, 1, 5, 10, 1024]),
        trigger_bounds_check=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_prefetch_pipeline(
        self,
        **kwargs: Any,
    ) -> None:
        self._test_cache_prefetch_pipeline(
            **kwargs,
            prefetch_stream=None,
        )

    @optests.dontGenerateOpCheckTests("Serial OOM")
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_github)
    @unittest.skipIf(*running_on_rocm)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        mixed=st.booleans(),
        weights_cache_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        stochastic_rounding=st.booleans(),
        gather_uvm_cache_stats=st.booleans(),
        mixed_B=st.booleans(),
        mpp_n_passes=st.sampled_from([None, 1, 6, 12]),
        mpp_min_size=st.sampled_from([None, 1, 5, 10, 1024]),
        trigger_bounds_check=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_prefetch_pipeline_stream_1(
        self,
        **kwargs: Any,
    ) -> None:
        self._test_cache_prefetch_pipeline(
            **kwargs,
            prefetch_location="before_fwd",
            prefetch_stream=torch.cuda.Stream(),
        )

    @optests.dontGenerateOpCheckTests("Serial OOM")
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_github)
    @unittest.skipIf(*running_on_rocm)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        mixed=st.booleans(),
        weights_cache_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        stochastic_rounding=st.booleans(),
        gather_uvm_cache_stats=st.booleans(),
        mixed_B=st.booleans(),
        mpp_n_passes=st.sampled_from([None, 1, 6, 12]),
        mpp_min_size=st.sampled_from([None, 1, 5, 10, 1024]),
        trigger_bounds_check=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_prefetch_pipeline_stream_2(
        self,
        **kwargs: Any,
    ) -> None:
        self._test_cache_prefetch_pipeline(
            **kwargs,
            prefetch_location="between_fwd_bwd",
            prefetch_stream=torch.cuda.Stream(),
        )

    @given(
        S=st.sampled_from([0, 7, 100, 1024]),
        mpp_n_passes=st.sampled_from([None, 1, 6, 12]),
        mpp_min_size=st.sampled_from([None, 1, 5, 10, 128]),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_get_prefetch_passes(
        self, S: int, mpp_n_passes: Optional[int], mpp_min_size: Optional[int]
    ) -> None:
        mpp_conf: Optional[MultiPassPrefetchConfig] = None
        if mpp_n_passes or mpp_min_size:
            mpp_conf = MultiPassPrefetchConfig()
            if mpp_n_passes:
                mpp_conf = mpp_conf._replace(num_passes=mpp_n_passes)
            if mpp_min_size:
                mpp_conf = mpp_conf._replace(min_splitable_pass_size=mpp_min_size)
        input_tensor = torch.randn(S)
        output_tensor = torch.randn(S)

        ret = SplitTableBatchedEmbeddingBagsCodegen.get_prefetch_passes(
            mpp_conf, input_tensor, output_tensor
        )

        if not mpp_conf:
            self.assertEqual(len(ret), 1)
            self.assertTrue(torch.equal(ret[0][0], input_tensor))
            self.assertTrue(torch.equal(ret[0][1], output_tensor))
            self.assertEqual(ret[0][2], 0)
            return

        # Make sure the max passes is not exceeding the configured value
        self.assertGreaterEqual(mpp_conf.num_passes, len(ret))

        # Make sure the passes are having the right start offset. Also make sure
        # every pass would not go below the configured min size (except for the
        # last pass)
        for idx, t in enumerate(ret):
            i, o, s = t
            if idx < len(ret) - 1:
                self.assertGreaterEqual(i.numel(), mpp_conf.min_splitable_pass_size)
            self.assertTrue(torch.equal(i, input_tensor[s : s + i.numel()]))
            self.assertTrue(torch.equal(o, output_tensor[s : s + i.numel()]))

        # Make sure the returned passes are both non-overlapping and complete. We do
        # this by settong the tensor to all zero, and increment them when visited
        input_tensor.zero_()
        output_tensor.zero_()
        for i, o, _ in ret:
            i.add_(1)
            o.add_(1)
        self.assertTrue(torch.equal(torch.full_like(input_tensor, 1), input_tensor))
        self.assertTrue(torch.equal(torch.full_like(output_tensor, 1), output_tensor))

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_github)
    @given(
        L=st.integers(min_value=0, max_value=16),
        H=st.integers(min_value=512, max_value=1024),
        S=st.integers(min_value=0, max_value=128),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_update_function(self, L: int, H: int, S: int) -> None:
        # Generate synthetic data
        linear_cache_indices_cpu = torch.randint(L, H, (S,))
        lxu_cache_locations_cpu = torch.clone(linear_cache_indices_cpu)

        indices = [True if np.random.rand() < 0.5 else False for _ in range(S)]
        lxu_cache_locations_cpu[indices] = -1

        cache_miss_ids = torch.clone(linear_cache_indices_cpu)
        cache_miss_ids[lxu_cache_locations_cpu != -1] = -2

        # Calculate the correct output
        unique_cache_miss_ids = torch.unique(cache_miss_ids)
        expect_out = sum(unique_cache_miss_ids >= 0)
        linear_cache_indices = to_device(
            torch.tensor(linear_cache_indices_cpu, dtype=torch.int64), use_cpu=False
        )
        lxu_cache_locations = to_device(
            torch.tensor(lxu_cache_locations_cpu, dtype=torch.int32), use_cpu=False
        )

        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        cc = emb_op(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.MANAGED_CACHING,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            record_cache_metrics=RecordCacheMetrics(True, False),
        )
        cc._update_cache_miss_counter(lxu_cache_locations, linear_cache_indices)
        (
            cache_miss_forward_count,
            unique_cache_miss_count,
        ) = cc.get_cache_miss_counter().cpu()

        self.assertEqual(unique_cache_miss_count, expect_out)
        self.assertLessEqual(cache_miss_forward_count, unique_cache_miss_count)

    def _get_unique_indices_reference(
        self,
        linear_indices: torch.Tensor,
        max_indices: int,
        compute_count: bool,
        compute_inverse_indices: bool,
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Python reference implementation for validating get_unique_indices operations.

        This function provides an independent baseline for testing CPU and GPU implementations
        of get_unique_indices. It uses only pure Python operations (sorted(), set(), dict).

        Example:
            Input:
                linear_indices = [20, 0, 10, 10, 0]
                max_indices = 20
                compute_count = True
                compute_inverse_indices = True
            Output:
                unique_indices = [0, 10, 20, x, x] where x is the uninitialized value.
                unique_indices_length = [3]
                unique_indices_count = [2, 2, 1, x, x]  (0 appears 2 times, 10 appears 2 times, 20 appears 1 time)
                linear_index_positions_sorted = [1, 4, 2, 3, 0]  (positions that sort the input: linear_indices[[1,4,2,3,0]] = [0,0,10,10,20])

        Args:
            linear_indices (Tensor): Input tensor of indices to find unique values from
            max_indices (int): Maximum possible index value (not used in computation, kept for API compatibility)
            compute_count (bool): If True, count occurrence for each unique index
            compute_inverse_indices (bool): If True, store original positions of the indices in a sorted manner

        Returns:
            A tuple containing:
            - unique_indices (Tensor): Tensor of size `linear_indices` that stores unique values in sorted order (i.e., unique values padded to input size)
            - unique_indices_length (Tensor): Tensor of size 1 containing number of unique values
            - unique_indices_count (Optional[Tensor]): If compute_count=True, tensor of size `linear_indices` that contains an occurrence count for each unique value, else None.
            - linear_index_positions_sorted (Optional[Tensor]): If compute_inverse_indices=True, tensor of size `linear_indices` that contains original positions such that linear_indices[linear_index_positions_sorted] produces sorted indices. Otherwise, None.
        """
        N = linear_indices.numel()

        # Convert to Python list for pure Python processing
        indices_list = linear_indices.tolist()

        # Get unique values in sorted order using pure Python
        unique_vals_list = sorted(set(indices_list))
        num_unique = len(unique_vals_list)

        # Prepare outputs matching the format of the ops
        unique_indices = torch.empty_like(linear_indices)
        if num_unique > 0:
            unique_indices[:num_unique] = torch.tensor(
                unique_vals_list, dtype=linear_indices.dtype
            )

        unique_indices_length = torch.tensor([num_unique], dtype=torch.int32)

        unique_indices_count = None
        if compute_count:
            # Count occurrences using pure Python
            count_dict = {}
            for val in indices_list:
                count_dict[val] = count_dict.get(val, 0) + 1

            counts_list = [count_dict[val] for val in unique_vals_list]

            unique_indices_count = torch.empty(N, dtype=torch.int32)
            if num_unique > 0:
                unique_indices_count[:num_unique] = torch.tensor(
                    counts_list, dtype=torch.int32
                )

        linear_index_positions_sorted = None
        if compute_inverse_indices:
            # Create list of (value, original_position) tuples
            indexed_list = [(val, idx) for idx, val in enumerate(indices_list)]

            # Sort by value (stable sort preserves order for equal values)
            sorted_indexed = sorted(indexed_list, key=lambda x: x[0])

            # Extract the original positions in sorted order
            positions_list = [pos for val, pos in sorted_indexed]

            linear_index_positions_sorted = torch.tensor(
                positions_list, dtype=torch.int32
            )

        return (
            unique_indices,
            unique_indices_length,
            unique_indices_count,
            linear_index_positions_sorted,
        )

    @given(
        N=st.integers(min_value=0, max_value=1000),
        max_indices=st.integers(min_value=100, max_value=10000),
        compute_count=st.booleans(),
        compute_inverse_indices=st.booleans(),
        dtype=st.sampled_from([torch.int, torch.long]),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_get_unique_indices_cpu(
        self,
        N: int,
        max_indices: int,
        compute_count: bool,
        compute_inverse_indices: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test get_unique_indices ops on CPU, GPU and MTIA.

        This test validates two ops:
        - torch.ops.fbgemm.get_unique_indices: Returns unique indices and optionally their counts
        - torch.ops.fbgemm.get_unique_indices_with_inverse: Additionally returns sorted positions for reordering

        The test uses a Python reference implementation (_get_unique_indices_reference) to ensure correctness and parity acorss devices.

        Test strategy:
        1. Generate random linear indices with values in [0, max_indices)
        2. Run pure Python reference implementation for ground truth
        3. Run CPU implementation via torch.ops.fbgemm.get_unique_indices[_with_inverse]
        4. Compare CPU results against reference implementation
        5. If GPU available, run the ops on GPU and compare against CPU results
        6. If MTIA available, run the ops on MTIA and compare against CPU results

        Validates:
        - Unique indices: Both CPU and GPU extract the same set of unique values in sorted order
        - Length: Number of unique values matches across all implementations
        - Counts (if compute_count=True): Occurrence count for each unique value matches
        - Positions (if compute_inverse_indices=True): Sorted positions produce identical reordering

        Args:
            N: Number of random linear indices to generate (0-1000). Tests with N=0 validate empty input handling.
            max_indices: Maximum value for generated indices (100-10000). Indices are in range [0, max_indices).
            compute_count: If True, ops return occurrence count for each unique value in the third output.
            compute_inverse_indices: If True, ops return original positions in sorted order (fourth output for
            get_unique_indices_with_inverse). These positions enable reordering the input to be in sorted order.
            dtype: Data type for generated indices. Tests both torch.int (int32) and torch.long (int64) to ensure
            CPU implementation supports all dtypes that CUDA implementation supports.
        """
        # Generate random linear indices with the specified dtype
        linear_indices = torch.randint(0, max_indices, (N,), dtype=dtype)

        # Get reference implementation results
        (
            unique_ref,
            length_ref,
            count_ref,
            positions_ref,
        ) = self._get_unique_indices_reference(
            linear_indices.cpu(), max_indices, compute_count, compute_inverse_indices
        )

        # Run on CPU
        if compute_inverse_indices:
            (
                unique_cpu,
                length_cpu,
                count_cpu,
                positions_cpu,
            ) = torch.ops.fbgemm.get_unique_indices_with_inverse(
                linear_indices,
                max_indices,
                compute_count,
                compute_inverse_indices,
            )
        else:
            unique_cpu, length_cpu, count_cpu = torch.ops.fbgemm.get_unique_indices(
                linear_indices, max_indices, compute_count
            )
            positions_cpu = None

        def compare_output(
            input_indices: torch.Tensor,
            annotate1: str,
            annotate2: str,
            length1: int,
            length2: int,
            unique1: torch.Tensor,
            unique2: torch.Tensor,
            compute_count: bool,
            compute_inverse_indices: bool,
            count1: Optional[torch.Tensor] = None,
            count2: Optional[torch.Tensor] = None,
            positions1: Optional[torch.Tensor] = None,
            positions2: Optional[torch.Tensor] = None,
        ):
            self.assertEqual(
                length1,
                length2,
                f"{annotate1} unique indices length mismatch with {annotate2}",
            )

            torch.testing.assert_close(
                unique1[:length1].cpu(),
                unique2[:length2].cpu(),
                msg=f"{annotate1} unique indices mismatch with {annotate2}",
            )

            if compute_count:
                self.assertIsNotNone(count1, f"{annotate1} count should not be None")
                self.assertIsNotNone(count2, f"{annotate2} count should not be None")
                torch.testing.assert_close(
                    count1[:length1].cpu(),
                    count2[:length2].cpu(),
                    msg=f"{annotate1} unique indices count mismatch with {annotate2}",
                )

            if compute_inverse_indices:
                self.assertIsNotNone(
                    positions1, f"{annotate1} positions should not be None"
                )
                self.assertIsNotNone(
                    positions2, f"{annotate2} positions should not be None"
                )

                torch.testing.assert_close(
                    positions1.cpu(),
                    positions2.cpu(),
                    msg=f"{annotate1} unique indices position mismatch with {annotate2}",
                )
                # Move positions to same device as input_indices before gather
                reordered1 = input_indices.gather(
                    0, positions1.long().to(input_indices.device)
                )
                reordered2 = input_indices.gather(
                    0, positions2.long().to(input_indices.device)
                )

                torch.testing.assert_close(
                    reordered1.cpu(),
                    reordered2.cpu(),
                    msg=f"{annotate1} reordered indices mismatch with {annotate2}",
                )

        # Test CPU op with reference
        compare_output(
            linear_indices,
            "CPU",
            "ref implementation",
            length_cpu.item(),
            length_ref,
            unique_cpu,
            unique_ref,
            compute_count,
            compute_inverse_indices,
            count_cpu,
            count_ref,
            positions_cpu,
            positions_ref,
        )

        # Run on GPU
        if not gpu_unavailable[0]:
            linear_indices_gpu = linear_indices.cuda()
            if compute_inverse_indices:
                (
                    unique_gpu,
                    length_gpu,
                    count_gpu,
                    positions_gpu,
                ) = torch.ops.fbgemm.get_unique_indices_with_inverse(
                    linear_indices_gpu,
                    max_indices,
                    compute_count,
                    compute_inverse_indices,
                )
            else:
                unique_gpu, length_gpu, count_gpu = torch.ops.fbgemm.get_unique_indices(
                    linear_indices_gpu, max_indices, compute_count
                )
                positions_gpu = None

            compare_output(
                linear_indices,
                "CPU",
                "GPU",
                length_cpu.item(),
                length_gpu.item(),
                unique_cpu,
                unique_gpu,
                compute_count,
                compute_inverse_indices,
                count_cpu,
                count_gpu,
                positions_cpu,
                positions_gpu,
            )

        # Run on MTIA
        if torch.mtia.is_available():
            linear_indices_mtia = linear_indices.mtia()
            if compute_inverse_indices:
                (
                    unique_mtia,
                    length_mtia,
                    count_mtia,
                    positions_mtia,
                ) = torch.ops.fbgemm.get_unique_indices_with_inverse(
                    linear_indices_mtia,
                    max_indices,
                    compute_count,
                    compute_inverse_indices,
                )
            else:
                unique_mtia, length_mtia, count_mtia = (
                    torch.ops.fbgemm.get_unique_indices(
                        linear_indices_gpu, max_indices, compute_count
                    )
                )
                positions_mtia = None

            compare_output(
                linear_indices,
                "CPU",
                "MTIA",
                length_cpu.item(),
                length_mtia.item(),
                unique_cpu,
                unique_mtia,
                compute_count,
                compute_inverse_indices,
                count_cpu,
                count_mtia,
                positions_cpu,
                positions_mtia,
            )

    @unittest.skipIf(*gpu_unavailable)
    @given(N=st.integers(min_value=1, max_value=8))
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_miss_counter(self, N: int) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        cc = emb_op(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.MANAGED_CACHING,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            record_cache_metrics=RecordCacheMetrics(True, True),
        )

        # Create fake input data and the target output
        xs = []
        x1 = torch.Tensor([[[1], [1]], [[3], [4]]])
        x1 = to_device(torch.tensor(x1, dtype=torch.int64), use_cpu=False)

        x2 = torch.Tensor([[[2], [1]], [[3], [4]]])
        x2 = to_device(torch.tensor(x2, dtype=torch.int64), use_cpu=False)

        x3 = torch.Tensor([[[5], [6]], [[7], [8]]])
        x3 = to_device(torch.tensor(x3, dtype=torch.int64), use_cpu=False)

        xs.append(x1)
        xs.append(x2)
        xs.append(x3)

        target_counter_list = [[1, 3], [2, 4], [3, 8]]
        target_tablewise_cache_miss_list = [[1, 2], [2, 2], [4, 4]]
        for x, t_counter, t_tablewise_cache_miss in zip(
            xs, target_counter_list, target_tablewise_cache_miss_list
        ):
            indices, offsets = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                cc(indices, offsets)
                (
                    cache_miss_forward_count,
                    unique_cache_miss_count,
                ) = cc.get_cache_miss_counter().cpu()
                tablewise_cache_miss = cc.get_table_wise_cache_miss().cpu()
                self.assertEqual(cache_miss_forward_count, t_counter[0])
                self.assertEqual(unique_cache_miss_count, t_counter[1])
                for i in range(len(tablewise_cache_miss)):
                    self.assertEqual(tablewise_cache_miss[i], t_tablewise_cache_miss[i])


if __name__ == "__main__":
    unittest.main()
