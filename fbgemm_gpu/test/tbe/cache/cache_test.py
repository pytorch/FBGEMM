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
from typing import Any, cast, List, Optional, Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType

from fbgemm_gpu.split_embedding_utils import (
    generate_requests,
    get_table_batched_offsets_from_dense,
    TBERequest,
    to_device,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
    RecordCacheMetrics,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from hypothesis import assume, given, settings

from ..common import MAX_EXAMPLES  # noqa E402

from .cache_common import (
    assert_cache,
    generate_cache_tbes,
    gpu_unavailable,
    optests,
    TestingStatsReporter,
    TestingStatsReporterConfig,
    VERBOSITY,
)


@optests.generate_opcheck_tests(fast=True)
class CacheTest(unittest.TestCase):
    def _compute_grad_output_shape(
        self,
        B: int,
        D_offsets: List[int],
        mixed_B: bool,
        Bs_feature_rank: Optional[List[List[int]]] = None,
    ) -> Tuple[int, ...]:
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
        # Need more than one table for variable batch sizes
        assume(not mixed_B or T > 1)
        assert prefetch_location in ["before_fwd", "between_fwd_bwd"]
        reporter = TestingStatsReporterConfig(interval=2)
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

        input_batch_count: List[int] = []
        intput_original_size: int = 0
        intput_long_size: int = 0
        output_batch_count: List[int] = []
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
        self.assertTrue(torch.all(cc.lxu_cache_locking_counter == 0))

        if prefetch_stream:
            # We record timing info at batch 1, 3, 5

            # But before, we need to wait until all backwards finished. Then
            # force report for async timer
            torch.cuda.synchronize()
            assert cc.bwd_wait_prefetch_timer, "Timer must have been set"
            cc.bwd_wait_prefetch_timer._lazy_report()

            self.assertIsInstance(cc.stats_reporter, TestingStatsReporter)
            stats_reporter: TestingStatsReporter = cast(
                TestingStatsReporter, cc.stats_reporter
            )

            def assert_event_exist(
                event_name: str,
                steps: List[int],
                expected_value: Optional[List[int]] = None,
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
                    self.assertEqual(rep_emb_id, "")
                    self.assertEqual(rep_tbe_id, "")

            def assert_event_not_exist(event_name: str) -> None:
                self.assertFalse(event_name in stats_reporter.reported_data)

            # Any reporting event happen before forward() will bear step timestamp
            # of 1 ~ 5, only odd step will be reported, so 1, 3, 5 steps will be in
            #
            # On the other side, if a reporting event happens after forward(), it'll
            # have step timestamp 0 ~ 4, so only 1, 3 steps will be in.
            assert_event_exist("bwd_wait_for_prefetch", [1, 3, 5], [])
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

    @unittest.skipIf(*gpu_unavailable)
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
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
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
