#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import unittest
from typing import Optional, Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType

from fbgemm_gpu.split_embedding_utils import (
    generate_requests,
    get_table_batched_offsets_from_dense,
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
    VERBOSITY,
)


@optests.generate_opcheck_tests(fast=True)
class CacheTest(unittest.TestCase):
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
    ) -> None:
        assume(weights_cache_precision == SparseType.FP16 or not stochastic_rounding)
        cc, cc_ref, min_Es, sum_Ds = generate_cache_tbes(
            T,
            D,
            log_E,
            mixed,
            cache_algorithm,
            weights_cache_precision=weights_cache_precision,
            stochastic_rounding=stochastic_rounding,
        )
        iters = 3
        requests = generate_requests(iters, B, T, L, min_Es, reuse=0.1)
        grad_output = torch.randn(B, sum_Ds).cuda()

        for indices, offsets, _ in requests:
            output = cc(indices, offsets)
            output_ref = cc_ref(indices, offsets)
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

        assert prefetch_location in ["before_fwd", "between_fwd_bwd"]
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
        )
        iters = 5
        requests = generate_requests(iters, B, T, L, min_Es, reuse=0.1)
        grad_output = (
            torch.randint(
                low=-10,
                high=10,
                size=(B, sum_Ds),
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
            batch: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        ) -> None:
            if not batch:
                return
            context_stream = prefetch_stream if prefetch_stream else cur_stream
            stream = cur_stream if prefetch_stream else None
            indices, offsets, _ = batch
            with torch.cuda.stream(context_stream):
                cc.prefetch(indices, offsets, stream)

        _prefetch(cc, batch_i)
        while batch_i:
            indices, offsets, _ = batch_i
            batch_ip1 = next(req_iter, None)
            if prefetch_stream:
                cur_stream.wait_stream(prefetch_stream)
            if prefetch_location == "before_fwd":
                _prefetch(cc, batch_ip1)
            output = cc(indices, offsets)
            if prefetch_location == "between_fwd_bwd":
                _prefetch(cc, batch_ip1)
            output.backward(grad_output)
            batch_i = batch_ip1
            batch_ip1 = None
        cc.flush()

        for indices, offsets, _ in requests:
            output_ref = cc_ref(indices, offsets)
            output_ref.backward(grad_output)

        for t in range(T):
            assert_cache(
                cc.split_embedding_weights()[t],
                cc_ref.split_embedding_weights()[t],
                stochastic_rounding,
            )

        assert_cache(output, output_ref, stochastic_rounding)
        self.assertTrue(torch.all(cc.lxu_cache_locking_counter == 0))

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
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_prefetch_pipeline(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
        prefetch_location: str,
        weights_cache_precision: SparseType,
        stochastic_rounding: bool,
    ) -> None:
        self._test_cache_prefetch_pipeline(
            T,
            D,
            B,
            log_E,
            L,
            mixed,
            prefetch_location,
            prefetch_stream=None,
            weights_cache_precision=weights_cache_precision,
            stochastic_rounding=stochastic_rounding,
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
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_prefetch_pipeline_stream_1(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
        weights_cache_precision: SparseType,
        stochastic_rounding: bool,
    ) -> None:
        self._test_cache_prefetch_pipeline(
            T,
            D,
            B,
            log_E,
            L,
            mixed,
            prefetch_location="before_fwd",
            prefetch_stream=torch.cuda.Stream(),
            weights_cache_precision=weights_cache_precision,
            stochastic_rounding=stochastic_rounding,
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
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_prefetch_pipeline_stream_2(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
        weights_cache_precision: SparseType,
        stochastic_rounding: bool,
    ) -> None:
        self._test_cache_prefetch_pipeline(
            T,
            D,
            B,
            log_E,
            L,
            mixed,
            prefetch_location="between_fwd_bwd",
            prefetch_stream=torch.cuda.Stream(),
            weights_cache_precision=weights_cache_precision,
            stochastic_rounding=stochastic_rounding,
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
