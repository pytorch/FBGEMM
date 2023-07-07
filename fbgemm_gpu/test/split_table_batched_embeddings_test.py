#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import copy
import math
import pickle
import random
import unittest
from itertools import accumulate
from typing import Any, List, Optional, Tuple, Union

import fbgemm_gpu
import hypothesis.strategies as st
import numpy as np
import torch

from fbgemm_gpu.split_embedding_configs import (
    EmbOptimType as OptimType,
    FP8QuantizationConfig,
    SparseType,
)
from fbgemm_gpu.split_embedding_optimizer_ops import (
    SplitEmbeddingArgs,
    SplitEmbeddingOptimizerParams,
    SplitEmbeddingRowwiseAdagrad,
)
from fbgemm_gpu.split_embedding_utils import (
    b_indices,
    fake_quantize_embs,
    generate_requests,
    get_table_batched_offsets_from_dense,
    quantize_embs,
    round_up,
    to_device,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
    RecordCacheMetrics,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
    rounded_row_size_in_bytes,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    CounterBasedRegularizationDefinition,
    CounterWeightDecayMode,
    DEFAULT_ASSOC,
    DenseTableBatchedEmbeddingBagsCodegen,
    GradSumDecay,
    INT8_EMB_ROW_DIM_OFFSET,
    LearningRateMode,
    SplitTableBatchedEmbeddingBagsCodegen,
    TailIdThreshold,
    WeightDecayMode,
)

from hypothesis import assume, given, HealthCheck, settings, Verbosity
from hypothesis.strategies import composite
from torch import Tensor

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_unavailable, TEST_WITH_ROCM
else:
    from fbgemm_gpu.test.test_utils import (
        gpu_available,
        gpu_unavailable,
        TEST_WITH_ROCM,
    )


MAX_EXAMPLES = 40

# For long running tests reduce the number of iterations to reduce timeout errors.
MAX_EXAMPLES_LONG_RUNNING = 15


@composite
# pyre-ignore
def get_nbit_weights_ty(draw) -> Optional[SparseType]:
    """
    Returns None if mixed weights ty should be used, otherwise, returns specific SparseType.
    """
    mixed_weights_ty = draw(st.booleans())
    if mixed_weights_ty:
        return None
    return draw(
        st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.FP8,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        )
    )


def gen_mixed_B_batch_sizes(B: int, T: int) -> Tuple[List[List[int]], List[int]]:
    num_ranks = np.random.randint(low=1, high=4)
    low = max(int(0.25 * B), 1)
    high = int(B)
    if low == high:
        Bs_rank_feature = [[B] * num_ranks for _ in range(T)]
    else:
        Bs_rank_feature = [
            np.random.randint(low=low, high=high, size=num_ranks).tolist()
            for _ in range(T)
        ]
    Bs = [sum(Bs_feature) for Bs_feature in Bs_rank_feature]
    return Bs_rank_feature, Bs


def format_ref_tensors_in_mixed_B_layout(
    ref_tensors: List[torch.Tensor], Bs_rank_feature: List[List[int]]
) -> torch.Tensor:
    # Relayout the reference tensor
    # Jagged dimension: (rank, table, local batch)
    num_ranks = len(Bs_rank_feature[0])
    split_tensors = [[] for _ in range(num_ranks)]  # shape (rank, table)
    for t, ref_tensor in enumerate(ref_tensors):
        assert ref_tensor.shape[0] == sum(Bs_rank_feature[t])
        tensors = ref_tensor.split(Bs_rank_feature[t])
        for r, tensor in enumerate(tensors):
            split_tensors[r].append(tensor.flatten())
    concat_list = []
    for r in range(num_ranks):
        concat_list += split_tensors[r]
    return torch.cat(concat_list, dim=0)


class SplitTableBatchedEmbeddingsTest(unittest.TestCase):
    def execute_forward_(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        mixed_B: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        pooling_mode: PoolingMode,
        use_cpu: bool,
        output_dtype: SparseType,
        use_experimental_tbe: bool,
    ) -> None:
        # NOTE: cache is not applicable to CPU version.
        assume(not use_cpu or not use_cache)
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        assume(not use_cpu or T * B * L * D <= 2048)
        # NOTE: CPU does not support FP16.
        assume(not (use_cpu and weights_precision == SparseType.FP16))

        # NOTE: weighted operation can be done only for SUM.
        assume(pooling_mode == PoolingMode.SUM or not weighted)
        # NOTE: No bag ops only work on GPUs, no mixed
        assume(not use_cpu or pooling_mode != PoolingMode.NONE)
        assume(not mixed or pooling_mode != PoolingMode.NONE)
        # TODO: Support these cases
        assume(
            not mixed_B
            or (
                weights_precision != SparseType.INT8
                and output_dtype != SparseType.INT8
                and not use_cpu
                and not use_cache
                and pooling_mode != PoolingMode.NONE
            )
        )

        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        if pooling_mode == PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
        elif pooling_mode == PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
        elif pooling_mode == PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
        else:
            # This proves that we have exhaustively checked all PoolingModes
            raise RuntimeError("Unknown PoolingMode!")

        E = int(10**log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]

        if not mixed_B:
            Bs = [B] * T
            Bs_rank_feature = [[0]]
        else:
            Bs_rank_feature, Bs = gen_mixed_B_batch_sizes(B, T)

        compute_device = ComputeDevice.CUDA
        if use_cpu:
            managed = [EmbeddingLocation.HOST] * T
            compute_device = ComputeDevice.CPU
        elif TEST_WITH_ROCM:
            # ROCm managed memory allocation is under development
            managed = [EmbeddingLocation.DEVICE] * T
        elif use_cache:
            managed = [EmbeddingLocation.MANAGED_CACHING] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        EmbeddingLocation.DEVICE if d < average_D else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        EmbeddingLocation.DEVICE,
                        EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        if do_pooling:
            bs = [
                to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]
        else:
            bs = [
                to_device(torch.nn.Embedding(E, D, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]
        if weights_precision == SparseType.INT8:
            for t in range(T):
                bs[t].weight.data.copy_(
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                            bs[t].weight.data
                        )
                    )
                )

        if weights_precision == SparseType.FP16:
            bs = [b.half() for b in bs]

        # Generate indices
        xs = [
            to_device(torch.randint(low=0, high=e, size=(b, L)), use_cpu)
            for e, b in zip(Es, Bs)
        ]
        # Generate positional weights
        xws = [to_device(torch.randn(size=(b, L)), use_cpu) for b in Bs]
        xws_acc_type = copy.deepcopy(xws)

        if weights_precision == SparseType.FP16:
            xws = [xw.half() for xw in xws]

        # Run baseline
        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, xs)
            ]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )

        if do_pooling:
            if mixed_B:
                f = format_ref_tensors_in_mixed_B_layout(fs, Bs_rank_feature)
            else:
                f = torch.cat([f.view(B, -1) for f in fs], dim=1)
        else:
            f = torch.cat(fs, dim=0).view(-1, D)

        # Create a TBE op
        cc = emb_op(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation(M),
                    compute_device,
                )
                for (E, D, M) in zip(Es, Ds, managed)
            ],
            weights_precision=weights_precision,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD
            if mixed_B
            else OptimType.EXACT_SGD,
            learning_rate=0.05,
            cache_algorithm=cache_algorithm,
            pooling_mode=pooling_mode,
            output_dtype=output_dtype,
            use_experimental_tbe=use_experimental_tbe,
        )
        # NOTE: test TorchScript-compatible!
        cc = torch.jit.script(cc)

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(
                bs[t].weight
                if weights_precision != SparseType.INT8
                else torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(bs[t].weight)
            )

        x = torch.cat([x.contiguous().flatten() for x in xs], dim=0)
        xw = torch.cat([xw.contiguous().flatten() for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(
            x, L, sum(Bs), use_cpu
        )

        batch_size_per_feature_per_rank = Bs_rank_feature if mixed_B else None

        # Run TBE
        fc2 = (
            cc(
                indices,
                offsets,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
            if not weighted
            else cc(
                indices,
                offsets,
                to_device(xw.contiguous().view(-1), use_cpu),
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
        )

        # Compare results: f = baseline, fc2 = TBE
        tolerance = (
            1.0e-5
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 8.0e-3
        )
        torch.testing.assert_close(
            fc2.float(), f.float(), atol=tolerance, rtol=tolerance
        )

    def test_forward_cpu_int8(
        self,
    ) -> None:
        weights_precision = SparseType.INT8
        use_cpu = True
        T = random.randint(1, 10)
        D = random.randint(2, min(256, int(2048 / T)))
        B = random.randint(1, min(128, int(2048 / T / D)))
        L = random.randint(0, min(20, int(2048 / T / D / B)))
        log_E = random.randint(3, 5)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
        )
        mixed = False
        mixed_B = False
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            SparseType.FP32,
            False,  # use_experimental_tbe
        )

    def test_forward_cpu_fp32(
        self,
    ) -> None:
        weights_precision = SparseType.FP32
        use_cpu = True
        T = random.randint(1, 10)
        D = random.randint(2, min(256, int(2048 / T)))
        B = random.randint(1, min(128, int(2048 / T / D)))
        L = random.randint(0, min(20, int(2048 / T / D / B)))
        log_E = random.randint(3, 5)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
        )
        mixed = False
        mixed_B = False
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            SparseType.FP32,
            False,  # use_experimental_tbe
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_forward_gpu_no_cache_int8(
        self,
    ) -> None:
        weights_precision = SparseType.INT8
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
        else:
            mixed = random.choice([True, False])
        mixed_B = False
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            SparseType.FP32,
            False,  # use_experimental_tbe
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        use_experimental_tbe=st.booleans() if not TEST_WITH_ROCM else st.just(False),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_gpu_no_cache_fp16(
        self,
        use_experimental_tbe: bool,
    ) -> None:
        weights_precision = SparseType.FP16
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
            + ([PoolingMode.NONE] if not use_experimental_tbe else [])
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
            mixed_B = False
        else:
            mixed = random.choice([True, False])
            mixed_B = (
                random.choice([True, False]) if not use_experimental_tbe else False
            )
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            SparseType.FP32,
            use_experimental_tbe,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        use_experimental_tbe=st.booleans() if not TEST_WITH_ROCM else st.just(False),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_gpu_no_cache_fp32(
        self,
        use_experimental_tbe: bool,
    ) -> None:
        weights_precision = SparseType.FP32
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
            + ([PoolingMode.NONE] if not use_experimental_tbe else [])
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
            mixed_B = False
        else:
            mixed = random.choice([True, False])
            mixed_B = (
                random.choice([True, False]) if not use_experimental_tbe else False
            )
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            SparseType.FP32,
            use_experimental_tbe,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        cache_algorithm=st.sampled_from(CacheAlgorithm),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_gpu_uvm_cache_int8(
        self,
        cache_algorithm: CacheAlgorithm,
    ) -> None:
        weights_precision = SparseType.INT8
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = True

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        )
        output_dtype = random.choice(
            [
                SparseType.FP32,
                SparseType.FP16,
            ]
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
        else:
            mixed = random.choice([True, False])
        mixed_B = False
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            output_dtype,
            False,  # use_experimental_tbe
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        use_experimental_tbe=st.booleans() if not TEST_WITH_ROCM else st.just(False),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_gpu_uvm_cache_fp16(
        self,
        cache_algorithm: CacheAlgorithm,
        use_experimental_tbe: bool,
    ) -> None:
        weights_precision = SparseType.FP16
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = True

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
            + ([PoolingMode.NONE] if not use_experimental_tbe else [])
        )
        output_dtype = random.choice(
            [
                SparseType.FP32,
                SparseType.FP16,
            ]
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
        else:
            mixed = random.choice([True, False])
        mixed_B = False
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            output_dtype,
            use_experimental_tbe,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        use_experimental_tbe=st.booleans() if not TEST_WITH_ROCM else st.just(False),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_gpu_uvm_cache_fp32(
        self,
        cache_algorithm: CacheAlgorithm,
        use_experimental_tbe: bool,
    ) -> None:
        weights_precision = SparseType.FP32
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = True

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
            + ([PoolingMode.NONE] if not use_experimental_tbe else [])
        )
        output_dtype = random.choice(
            [
                SparseType.FP32,
                SparseType.FP16,
            ]
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
        else:
            mixed = random.choice([True, False])
        mixed_B = False
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            output_dtype,
            use_experimental_tbe,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        output_dtype=st.sampled_from([SparseType.FP16, SparseType.INT8]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_forward_fused_pooled_emb_quant(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        output_dtype: SparseType,
    ) -> None:
        Ds = [
            round_up(np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)), 4)
            for _ in range(T)
        ]
        E = int(10**log_E)
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]

        op = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.DEVICE,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            output_dtype=output_dtype,
            device=torch.cuda.current_device(),
        )
        op_ref = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.DEVICE,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            output_dtype=SparseType.FP32,
            device=torch.cuda.current_device(),
        )
        # sync weights between two ops
        split_weights = op.split_embedding_weights()
        ref_split_weights = op_ref.split_embedding_weights()
        for t in range(T):
            split_weights[t].data.copy_(ref_split_weights[t])

        requests = generate_requests(2, B, T, L, min(Es), reuse=0.1)

        for indices, offsets, _ in requests:
            lowp_pooled_output = op(
                indices=indices,
                offsets=offsets,
            )
            fp32_pooled_output = op_ref(
                indices=indices,
                offsets=offsets,
            )
            lowp_pooled_emb_split = [
                d + 8 if output_dtype == SparseType.INT8 else d for d in op.dims
            ]
            lowp_pooled_output_per_table = torch.split(
                lowp_pooled_output, lowp_pooled_emb_split, dim=1
            )
            deq_lowp_pooled_output_per_table = [
                torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(t.contiguous())
                if output_dtype == SparseType.INT8
                else t.float()
                for t in lowp_pooled_output_per_table
            ]
            fp32_pooled_output_per_table = torch.split(
                fp32_pooled_output, op.dims, dim=1
            )
            dq_fp32_pooled_output_per_table = [
                torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                    torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                        t.contiguous()
                    ).contiguous()
                )
                if output_dtype == SparseType.INT8
                else t.half().float()
                for t in fp32_pooled_output_per_table
            ]
            cat_deq_lowp_pooled_output = torch.cat(
                deq_lowp_pooled_output_per_table, dim=1
            )
            cat_dq_fp32_pooled_output = torch.cat(
                dq_fp32_pooled_output_per_table, dim=1
            )
            torch.testing.assert_close(
                cat_deq_lowp_pooled_output, cat_dq_fp32_pooled_output
            )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                # FIXME: INT2 caused big numerical error for this test
                # SparseType.INT2,
            ]
        ),
        output_dtype=st.sampled_from(
            [
                SparseType.FP16,
                SparseType.BF16,
                SparseType.INT8,
                # SparseType.INT4,
            ]
        )
        if not TEST_WITH_ROCM
        else st.sampled_from(
            [
                SparseType.FP16,
                # The counterparts of __nv_bfloat16 and __nv_bfloat162 are not supported on ROCm
                SparseType.INT8,
                # SparseType.INT4,
            ]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_nbit_forward_fused_pooled_emb_quant(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_ty: SparseType,
        output_dtype: SparseType,
    ) -> None:
        D_alignment = max(weights_ty.align_size() for t in range(T))
        D_alignment = max(D_alignment, output_dtype.align_size())
        D = round_up(D, D_alignment)
        # BF16 output only works for CUDA device sm80+ (e.g., A100)
        assume(
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (8, 0)
            or not output_dtype == SparseType.BF16
        )
        Ds = [
            round_up(
                np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                D_alignment,
            )
            for _ in range(T)
        ]
        Ds = [D] * T
        E = int(10**log_E)
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]

        weights_ty_list = [weights_ty] * T
        managed = [EmbeddingLocation.DEVICE] * T
        op = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    W_TY,
                    EmbeddingLocation(M),
                )
                for (E, D, M, W_TY) in zip(Es, Ds, managed, weights_ty_list)
            ],
            output_dtype=output_dtype,
            device=torch.cuda.current_device(),
        )
        # Initialize the random weights for int nbit table split embedding bag
        op.fill_random_weights()

        op_ref = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    W_TY,
                    EmbeddingLocation(M),
                )
                for (E, D, M, W_TY) in zip(Es, Ds, managed, weights_ty_list)
            ],
            output_dtype=SparseType.FP32,
            device=torch.cuda.current_device(),
        )
        # Initialize the random weights for int nbit table split embedding bag
        op_ref.fill_random_weights()

        # sync weights between two ops
        split_weights = op.split_embedding_weights()
        ref_split_weights = op_ref.split_embedding_weights()
        for t in range(T):
            (weights, scale_shift) = split_weights[t]
            (ref_weights, ref_scale_shift) = ref_split_weights[t]
            self.assertEqual(weights.size(), ref_weights.size())
            element_size = weights_ty_list[t].bit_rate() / 8.0
            rand_tensor = torch.rand(
                ref_weights.shape[0], int(ref_weights.shape[1] / element_size)
            )
            rand_weights, rand_scale_shift = quantize_embs(
                rand_tensor, weights_ty_list[t]
            )
            ref_weights.copy_(rand_weights)
            weights.copy_(ref_weights)
            if rand_scale_shift is not None:
                self.assertIsNotNone(scale_shift)
                self.assertIsNotNone(ref_scale_shift)
                ref_scale_shift.copy_(rand_scale_shift)
                scale_shift.copy_(ref_scale_shift)

        requests = generate_requests(1, B, T, L, min(Es), reuse=0.1)
        for indices, offsets, _ in requests:
            lowp_pooled_output = op(
                indices=indices.int(),
                offsets=offsets.int(),
            )
            fp32_pooled_output = op_ref(
                indices=indices.int(),
                offsets=offsets.int(),
            )
            lowp_pooled_emb_split = [
                d + 8 if output_dtype == SparseType.INT8 else d for d in Ds
            ]
            lowp_pooled_output_per_table = torch.split(
                lowp_pooled_output, lowp_pooled_emb_split, dim=1
            )
            deq_lowp_pooled_output_per_table = [
                torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(t.contiguous())
                if output_dtype == SparseType.INT8
                else t.float()
                for t in lowp_pooled_output_per_table
            ]
            fp32_pooled_output_per_table = torch.split(fp32_pooled_output, Ds, dim=1)
            dq_fp32_pooled_output_per_table = [
                torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                    torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                        t.contiguous()
                    ).contiguous()
                ).contiguous()
                if output_dtype == SparseType.INT8
                else t.half().float()
                for t in fp32_pooled_output_per_table
            ]
            cat_deq_lowp_pooled_output = torch.cat(
                deq_lowp_pooled_output_per_table, dim=1
            )
            cat_dq_fp32_pooled_output = torch.cat(
                dq_fp32_pooled_output_per_table, dim=1
            )
            torch.testing.assert_close(
                cat_deq_lowp_pooled_output,
                cat_dq_fp32_pooled_output,
                rtol=1e-2,
                atol=1e-2,
                equal_nan=True,
            )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        ),
        output_dtype=st.sampled_from(
            [
                SparseType.FP16,
                SparseType.BF16,
                SparseType.INT8,
            ]
        )
        if not TEST_WITH_ROCM
        else st.sampled_from(
            [
                SparseType.FP16,
                # The counterparts of __nv_bfloat16 and __nv_bfloat162 are not supported on ROCm
                SparseType.INT8,
            ]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_nbit_split_embedding_weights_with_scale_and_bias(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_ty: SparseType,
        output_dtype: SparseType,
    ) -> None:
        D_alignment = max(weights_ty.align_size() for t in range(T))
        D_alignment = max(D_alignment, output_dtype.align_size())
        D = round_up(D, D_alignment)
        # BF16 output only works for CUDA device sm80+ (e.g., A100)
        assume(
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (8, 0)
            or not output_dtype == SparseType.BF16
        )
        Ds = [
            round_up(
                np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                D_alignment,
            )
            for _ in range(T)
        ]
        Ds = [D] * T
        E = int(10**log_E)
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]

        weights_ty_list = [weights_ty] * T
        managed = [EmbeddingLocation.DEVICE] * T
        op = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    W_TY,
                    EmbeddingLocation(M),
                )
                for (E, D, M, W_TY) in zip(Es, Ds, managed, weights_ty_list)
            ],
            output_dtype=output_dtype,
            device=torch.cuda.current_device(),
        )
        # Initialize the random weights for int nbit table split embedding bag
        op.fill_random_weights()

        # sync weights between two ops
        split_weights = op.split_embedding_weights()
        split_weights_with_scale_bias = op.split_embedding_weights_with_scale_bias(
            split_scale_bias_mode=2
        )
        for t in range(T):
            (weights, scale_bias) = split_weights[t]
            (weights2, scale, bias) = split_weights_with_scale_bias[t]
            torch.testing.assert_close(weights2, weights)
            if scale is None:
                self.assertIsNone(scale_bias)
                self.assertIsNone(bias)
            else:
                torch.testing.assert_close(
                    scale.cpu(),
                    torch.tensor(
                        scale_bias[:, : scale_bias.size(1) // 2]
                        .contiguous()
                        .cpu()
                        .numpy()
                        .view(np.float16)
                    ),
                )
                torch.testing.assert_close(
                    bias.cpu(),
                    torch.tensor(
                        scale_bias[:, scale_bias.size(1) // 2 :]
                        .contiguous()
                        .cpu()
                        .numpy()
                        .view(np.float16)
                    ),
                )

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=32),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=10),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_dense(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        long_segments: bool,
        pooling_mode: PoolingMode,
        use_cpu: bool,
        output_dtype: SparseType,
    ) -> None:
        # NOTE: torch.autograd.gradcheck() is too time-consuming for CPU version
        #       so we have to limit (T * B * L * D)!
        assume(not use_cpu or T * B * L * D <= 2048)
        assume(pooling_mode == PoolingMode.SUM or not weighted)
        assume(not (use_cpu and weights_precision == SparseType.FP16))
        # No bag ops only work on GPUs, no mixed, no weighted
        assume(not use_cpu or pooling_mode != PoolingMode.NONE)
        assume(not mixed or pooling_mode != PoolingMode.NONE)
        assume(not weighted or pooling_mode != PoolingMode.NONE)

        emb_op = DenseTableBatchedEmbeddingBagsCodegen
        if pooling_mode == PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
        elif pooling_mode == PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
        elif pooling_mode == PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
        else:
            # This proves that we have exhaustively checked all PoolingModes
            raise RuntimeError("Unknown PoolingMode!")

        E = int(10**log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2 * E)) for _ in range(T)
            ]
        if do_pooling:
            bs = [
                to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=False), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]
        else:
            bs = [
                to_device(torch.nn.Embedding(E, D, sparse=False), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]

        if weights_precision == SparseType.FP16:
            bs = [b.half() for b in bs]

        xs = [
            to_device(
                torch.from_numpy(
                    np.random.choice(range(e), size=(B, L), replace=True).astype(
                        np.int64
                    )
                ),
                use_cpu,
            )
            for e in Es
        ]
        if long_segments and L > 0 and weights_precision != SparseType.FP16:
            for x in xs:
                x[:, 0] = 0

        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(T)]
        xws_acc_type = copy.deepcopy(xws)

        if weights_precision == SparseType.FP16:
            xws = [xw.half() for xw in xws]

        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, xs)
            ]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        gos = [torch.randn_like(f) for f in fs]
        [f.backward(go) for (f, go) in zip(fs, gos)]

        # pyre-fixme[16]: `Optional` has no attribute `view`.
        grad_weights = torch.cat([b.weight.grad.view(-1) for b in bs])
        if weights_precision == SparseType.FP16 and not use_cpu:
            grad_weights = grad_weights.half()

        cc = emb_op(
            embedding_specs=[(E, D) for (E, D) in zip(Es, Ds)],
            pooling_mode=pooling_mode,
            use_cpu=use_cpu,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
        )
        if do_pooling:
            # NOTE: test TorchScript-compatible!
            cc = torch.jit.script(cc)

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=use_cpu)
        fc2 = (
            cc(indices, offsets)
            if not weighted
            else cc(indices, offsets, to_device(xw.contiguous().view(-1), use_cpu))
        )

        if do_pooling:
            f = torch.cat([f.view(B, -1) for f in fs], dim=1)
        else:
            f = torch.cat(fs, dim=0).view(-1, D)

        torch.testing.assert_close(
            fc2.float(),
            f.float(),
            atol=5.0e-3
            if weights_precision == SparseType.FP16 or output_dtype == SparseType.FP16
            else 1.0e-5,
            rtol=5.0e-3
            if weights_precision == SparseType.FP16 or output_dtype == SparseType.FP16
            else 1.0e-5,
        )
        if do_pooling:
            goc = torch.cat([go.view(B, -1) for go in gos], dim=1)
        else:
            goc = torch.cat(gos, dim=0)
        fc2.backward(goc)
        torch.testing.assert_close(
            cc.weights.grad,
            grad_weights,
            atol=5.0e-3
            if weights_precision == SparseType.FP16 or output_dtype == SparseType.FP16
            else 1.0e-4,
            rtol=5.0e-3
            if weights_precision == SparseType.FP16 or output_dtype == SparseType.FP16
            else 1.0e-4,
        )

        cc = DenseTableBatchedEmbeddingBagsCodegen(
            [(E, D) for (E, D) in zip(Es, Ds)],
            # NOTE: only SUM pooling can work with per_sample_weights!
            pooling_mode=PoolingMode.SUM,
            use_cpu=use_cpu,
        )

        per_sample_weights = to_device(xw.contiguous().view(-1), use_cpu)
        if use_cpu:
            # NOTE: GPU version of DenseTableBatchedEmbeddingBagsCodegen doesn't support double.
            cc = cc.double()
            per_sample_weights = per_sample_weights.double()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        for param in cc.parameters():
            param.requires_grad = False
        y = cc(indices, offsets, per_sample_weights)
        y.sum().backward()
        # pyre-fixme[16]: `Optional` has no attribute `clone`.
        indice_weight_grad_all = per_sample_weights.grad.clone().cpu()
        T_ = len(xws)
        feature_requires_grad = to_device(
            torch.tensor(np.random.choice([0, 1], replace=True, size=(T_,))).int(),
            use_cpu,
        )
        per_sample_weights = per_sample_weights.detach().clone()
        per_sample_weights.requires_grad = True
        y = cc(
            indices,
            offsets,
            per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        )
        y.sum().backward()
        indice_weight_grad_mask = per_sample_weights.grad.clone().cpu()
        for t in range(T_):
            if feature_requires_grad[t]:
                torch.testing.assert_close(
                    indice_weight_grad_mask.view(T_, B, L)[t],
                    indice_weight_grad_all.view(T_, B, L)[t],
                )
            else:
                torch.testing.assert_close(
                    indice_weight_grad_mask.view(T_, B, L)[t],
                    torch.zeros_like(indice_weight_grad_mask.view(T_, B, L)[t]),
                )

        per_sample_weights = to_device(xw.contiguous().view(-1), use_cpu)
        if use_cpu:
            # NOTE: GPU version of DenseTableBatchedEmbeddingBagsCodegen doesn't support double.
            cc = cc.double()
            per_sample_weights = per_sample_weights.double()
        else:
            cc = cc.float()
            per_sample_weights = per_sample_weights.float()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        for param in cc.parameters():
            param.requires_grad = False
        torch.autograd.gradcheck(
            cc, (indices, offsets, per_sample_weights), eps=1e-2, atol=1e-3, rtol=1e-3
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        ),
        output_dtype=st.sampled_from([SparseType.FP16, SparseType.FP32]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_none(self, **kwargs: Any) -> None:
        self.execute_backward_none_(**kwargs)

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        ),
        output_dtype=st.sampled_from([SparseType.FP16, SparseType.FP32]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_none_with_rowwise_adagrad(self, **kwargs: Any) -> None:
        self.execute_backward_none_(optimizer=OptimType.EXACT_ROWWISE_ADAGRAD, **kwargs)

    def execute_backward_none_(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        long_segments: bool,
        pooling_mode: PoolingMode,
        output_dtype: SparseType,
        optimizer: Optional[OptimType] = None,
    ) -> None:
        use_cpu = False
        mixed = False
        use_cache = False

        # NOTE: cache is not applicable to CPU version.
        assume(not use_cpu or not use_cache)
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        assume(not use_cpu or T * B * L * D <= 2048)
        assume(not (use_cpu and weights_precision == SparseType.FP16))
        # No bag ops only work on GPUs, no mixed, no weighted
        assume(not use_cpu or pooling_mode != PoolingMode.NONE)
        assume(not mixed or pooling_mode != PoolingMode.NONE)
        assume(not weighted or pooling_mode != PoolingMode.NONE)

        assume(pooling_mode == PoolingMode.SUM or not weighted)

        if pooling_mode == PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
        elif pooling_mode == PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
        elif pooling_mode == PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
        else:
            # This proves that we have exhaustively checked all PoolingModes
            raise RuntimeError("Unknown PoolingMode!")

        E = int(10**log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        compute_device = ComputeDevice.CUDA
        if use_cpu:
            managed = [EmbeddingLocation.HOST] * T
            compute_device = ComputeDevice.CPU
        elif TEST_WITH_ROCM:
            # ROCm managed memory allocation is under development
            managed = [EmbeddingLocation.DEVICE] * T
        elif use_cache:
            managed = [EmbeddingLocation.MANAGED_CACHING] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        EmbeddingLocation.DEVICE if d < average_D else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        EmbeddingLocation.DEVICE,
                    ]
                )
                for _ in range(T)
            ]
        if do_pooling:
            bs = [
                to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]
        else:
            bs = [
                to_device(torch.nn.Embedding(E, D, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]

        if weights_precision == SparseType.FP16:
            bs = [b.half() for b in bs]

        feature_table_map = list(range(T))
        xs = [
            to_device(
                torch.from_numpy(
                    np.random.choice(range(Es[t]), size=(B, L)).astype(np.int64)
                ),
                use_cpu,
            )
            for t in feature_table_map
        ]

        if long_segments and L > 0:
            for x in xs:
                x[:, 0] = 0

        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(len(xs))]
        xws_acc_type = copy.deepcopy(xws)

        if weights_precision == SparseType.FP16:
            xws = [xw.half() for xw in xws]

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=use_cpu)
        embedding_specs = [
            (E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)
        ]

        # Hyperparameters in case optimizer is not None
        lr = 0.5
        eps = 0.2
        stochastic_rounding = random.choice([True, False])

        if optimizer is None:
            fs = (
                [
                    b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                    for (b, x) in zip(bs, xs)
                ]
                if not weighted
                else [
                    b_indices(
                        b,
                        x,
                        per_sample_weights=xw.view(-1),
                        use_cpu=use_cpu,
                        do_pooling=do_pooling,
                    )
                    for (b, x, xw) in zip(bs, xs, xws)
                ]
            )
            gos: Union[List[Tensor], Tensor] = [torch.randn_like(f) for f in fs]
            [f.backward(go) for (f, go) in zip(fs, gos)]
        else:
            bs_ = SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=embedding_specs,
                optimizer=optimizer,
                feature_table_map=feature_table_map,
                weights_precision=weights_precision,
                pooling_mode=pooling_mode,
                output_dtype=output_dtype,
                learning_rate=lr,
                eps=eps,
                stochastic_rounding=stochastic_rounding,
            )

            for t in range(T):
                bs_.split_embedding_weights()[t].data.copy_(bs[t].weight)

            fs = (
                bs_(indices, offsets)
                if not weighted
                else bs_(
                    indices,
                    offsets,
                    to_device(xw.contiguous().view(-1), use_cpu),
                )
            )
            gos: Union[List[Tensor], Tensor] = torch.rand_like(fs)
            fs.backward(gos)

        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            optimizer=OptimType.NONE,
            feature_table_map=feature_table_map,
            weights_precision=weights_precision,
            pooling_mode=pooling_mode,
            output_dtype=output_dtype,
        )

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        total_unique_indices = 0
        # Compute number of unique indices
        for t in range(len(feature_table_map)):
            start = offsets[t * B]
            end = offsets[(t + 1) * B]
            uniq_indices = indices[start:end].unique()
            total_unique_indices += uniq_indices.numel()

        fc2 = (
            cc(indices, offsets, total_unique_indices=total_unique_indices)
            if not weighted
            else cc(
                indices,
                offsets,
                to_device(xw.contiguous().view(-1), use_cpu),
                total_unique_indices=total_unique_indices,
            )
        )
        if optimizer is None:
            assert type(gos) == list
            if do_pooling:
                goc = torch.cat([go.view(B, -1) for go in gos], dim=1)
            else:
                goc = torch.cat(gos, dim=0)
        else:
            assert type(gos) == Tensor
            goc = gos.clone()
        fc2.backward(goc)

        if optimizer is not None:
            # pyre-ignore[6]
            params = SplitEmbeddingOptimizerParams(weights_dev=cc.weights_dev)
            embedding_args = SplitEmbeddingArgs(
                # pyre-ignore[6]
                weights_placements=cc.weights_placements,
                # pyre-ignore[6]
                weights_offsets=cc.weights_offsets,
                max_D=cc.max_D,
            )
            optim = SplitEmbeddingRowwiseAdagrad(
                params,
                embedding_args,
                embedding_specs,
                feature_table_map,
                learning_rate=lr,
                eps=eps,
                stochastic_rounding=stochastic_rounding,
            )
            optim.step()

        if use_cache:
            cc.flush()

        if optimizer is None:
            test_tensor = cc.weights_dev.grad
            weight_grads = []
            for t in range(T):
                grad = bs[t].weight.grad
                # Check grad to suppress pyre error
                assert grad is not None
                weight_grads.append(grad)
            ref_grad = torch.concat(weight_grads, dim=0).to_sparse().coalesce()
            ref_tensor = (
                ref_grad.half() if weights_precision == SparseType.FP16 else ref_grad
            )
        else:
            # pyre-ignore[16]
            indices = cc.weights_dev.grad._indices().flatten()
            # Select only the part in the table that is updated
            test_tensor = torch.index_select(cc.weights_dev.view(-1, D), 0, indices)
            ref_tensor = torch.index_select(bs_.weights_dev.view(-1, D), 0, indices)

        tolerance = (
            1.0e-2
            if long_segments
            else (
                1.0e-4
                if weights_precision == SparseType.FP32
                and output_dtype == SparseType.FP32
                else 1.0e-2
            )
        )
        torch.testing.assert_close(
            test_tensor,
            ref_tensor,
            atol=tolerance,
            rtol=tolerance,
        )

    def execute_backward_sgd_(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        mixed_B: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        long_segments: bool,
        pooling_mode: PoolingMode,
        use_cpu: bool,
        output_dtype: SparseType,
    ) -> None:
        # NOTE: cache is not applicable to CPU version.
        assume(not use_cpu or not use_cache)
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        assume(not use_cpu or T * B * L * D <= 2048)
        assume(not (use_cpu and weights_precision == SparseType.FP16))
        # No bag ops only work on GPUs, no mixed, no weighted
        assume(not use_cpu or pooling_mode != PoolingMode.NONE)
        assume(not mixed or pooling_mode != PoolingMode.NONE)
        assume(not weighted or pooling_mode != PoolingMode.NONE)

        assume(pooling_mode == PoolingMode.SUM or not weighted)
        # TODO: Support these cases
        assume(
            not mixed_B
            or (
                weights_precision != SparseType.INT8
                and output_dtype != SparseType.INT8
                and not use_cpu
                and not use_cache
                and pooling_mode != PoolingMode.NONE
            )
        )

        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        if pooling_mode == PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
        elif pooling_mode == PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
        elif pooling_mode == PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
        else:
            # This proves that we have exhaustively checked all PoolingModes
            raise RuntimeError("Unknown PoolingMode!")

        E = int(10**log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]

        if not mixed_B:
            Bs = [B] * T
        else:
            low = max(int(0.25 * B), 1)
            high = int(B)
            if low == high:
                Bs = [B] * T
            else:
                Bs = [np.random.randint(low=low, high=high) for _ in range(T)]

        compute_device = ComputeDevice.CUDA
        if use_cpu:
            managed = [EmbeddingLocation.HOST] * T
            compute_device = ComputeDevice.CPU
        elif TEST_WITH_ROCM:
            # ROCm managed memory allocation is under development
            managed = [EmbeddingLocation.DEVICE] * T
        elif use_cache:
            managed = [EmbeddingLocation.MANAGED_CACHING] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        EmbeddingLocation.DEVICE if d < average_D else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        EmbeddingLocation.DEVICE,
                        EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        if do_pooling:
            bs = [
                to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]
        else:
            bs = [
                to_device(torch.nn.Embedding(E, D, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]

        if weights_precision == SparseType.FP16:
            bs = [b.half() for b in bs]

        feature_table_map = list(range(T))
        table_to_replicate = T // 2
        # pyre-fixme[6]: For 2nd param expected `Embedding` but got
        #  `Union[Embedding, EmbeddingBag]`.
        bs.insert(table_to_replicate, bs[table_to_replicate])
        feature_table_map.insert(table_to_replicate, table_to_replicate)

        num_features = len(feature_table_map)
        if not mixed_B:
            Bs = [B] * num_features
            Bs_rank_feature = [[0]]
        else:
            Bs_rank_feature, Bs = gen_mixed_B_batch_sizes(B, num_features)

        # Generate indices
        xs = [
            to_device(
                torch.from_numpy(
                    np.random.choice(range(Es[t]), size=(b, L), replace=True).astype(
                        np.int64
                    )
                ),
                use_cpu,
            )
            for t, b in zip(feature_table_map, Bs)
        ]

        if long_segments and L > 0:
            for x in xs:
                x[:, 0] = 0

        # Generate positional weights
        xws = [to_device(torch.randn(size=(b, L)), use_cpu) for b in Bs]
        xws_acc_type = copy.deepcopy(xws)

        if weights_precision == SparseType.FP16:
            xws = [xw.half() for xw in xws]

        # Run baseline's forward
        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, xs)
            ]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        # Generate gradients
        gos = [torch.randn_like(f) for f in fs]
        # Run baseline's backward
        [f.backward(go) for (f, go) in zip(fs, gos)]
        # do SGD update
        lr = 0.05
        del bs[table_to_replicate]
        # pyre-fixme[58]: `*` is not supported for operand types
        #  `Optional[torch._tensor.Tensor]` and `float`.
        new_weights = [(b.weight - b.weight.grad * lr) for b in bs]

        # Create a TBE op
        cc = emb_op(
            embedding_specs=[
                (E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)
            ],
            optimizer=OptimType.EXACT_SGD,
            feature_table_map=feature_table_map,
            learning_rate=lr,
            weights_precision=weights_precision,
            cache_algorithm=cache_algorithm,
            pooling_mode=pooling_mode,
            output_dtype=output_dtype,
        )

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.contiguous().flatten() for x in xs], dim=0)
        xw = torch.cat([xw.contiguous().flatten() for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(
            x, L, sum(Bs), use_cpu=use_cpu
        )

        batch_size_per_feature_per_rank = Bs_rank_feature if mixed_B else None

        # Run TBE's forward
        fc2 = (
            cc(
                indices,
                offsets,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
            if not weighted
            else cc(
                indices,
                offsets,
                to_device(xw.contiguous().view(-1), use_cpu),
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
        )
        # Generate gradients
        if do_pooling:
            if mixed_B:
                goc = format_ref_tensors_in_mixed_B_layout(gos, Bs_rank_feature)
            else:
                goc = torch.cat([go.view(B, -1) for go in gos], dim=1)
        else:
            goc = torch.cat(gos, dim=0)

        # Run TBE's backward
        fc2.backward(goc)

        if use_cache:
            cc.flush()
        for t in range(T):
            torch.testing.assert_close(
                cc.split_embedding_weights()[t],
                new_weights[t].half()
                if weights_precision == SparseType.FP16 and not use_cpu
                else new_weights[t],
                atol=1.0e-2
                if long_segments
                else (5.0e-3 if weights_precision == SparseType.FP16 else 1.0e-5),
                rtol=1.0e-1
                if long_segments
                else (2.0e-2 if weights_precision == SparseType.FP16 else 1.0e-5),
            )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_sgd(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        long_segments: bool,
        pooling_mode: PoolingMode,
        use_cpu: bool,
    ) -> None:
        self.execute_backward_sgd_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            False,  # mixed_B
            use_cache,
            cache_algorithm,
            long_segments,
            pooling_mode,
            use_cpu,
            SparseType.FP32,  # output_dtype
        )

    @given(
        D=st.integers(min_value=2, max_value=10),
        # 128 * 1024 is to exercise a case num_ctas_for_run needs to be capped
        # at the number of SMs (H100 SXM5 has 132 SMs and the default seglen
        # per CTA is 1024)
        B=st.sampled_from([1152, 256 * 1024]),
        L=st.integers(min_value=1, max_value=4),
        weighted=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_sgd_really_long_segments(  # noqa C901
        self,
        D: int,
        B: int,
        L: int,
        weighted: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
    ) -> None:
        self.execute_backward_sgd_(
            2,  # T
            D,
            B,
            1,  # log_E,
            L,
            SparseType.FP32,  # weights_precision
            weighted,
            mixed,
            False,  # mixed_B
            use_cache,
            cache_algorithm,
            True,  # long_segments
            PoolingMode.SUM,  # pooling_mode
            False,  # use_cpu
            SparseType.FP32,  # output_dtype
        )

    def execute_backward_adagrad_(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        D_gradcheck: int,
        weights_precision: SparseType,
        stochastic_rounding: bool,
        weighted: bool,
        row_wise: bool,
        mixed: bool,
        mixed_B: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        pooling_mode: PoolingMode,
        use_cpu: bool,
        output_dtype: SparseType,
        weight_decay_mode: WeightDecayMode = WeightDecayMode.NONE,
    ) -> None:
        # NOTE: cache is not applicable to CPU version.
        assume(not use_cpu or not use_cache)

        # NOTE: torch.autograd.gradcheck() is too time-consuming for CPU version
        #       so we have to limit (T * B * L * D)!
        assume(not use_cpu or T * B * L * D <= 1024)
        assume(not (use_cpu and weights_precision == SparseType.FP16))

        assume(
            pooling_mode == PoolingMode.SUM or not weighted
        )  # No bag ops only work on GPUs, no mixed, no weighted
        assume(not use_cpu or pooling_mode != PoolingMode.NONE)
        assume(not mixed or pooling_mode != PoolingMode.NONE)
        assume(not weighted or pooling_mode != PoolingMode.NONE)
        # TODO: Support these cases
        assume(
            not mixed_B
            or (
                weights_precision != SparseType.INT8
                and output_dtype != SparseType.INT8
                and not use_cpu
                and not use_cache
                and pooling_mode != PoolingMode.NONE
            )
        )

        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        if pooling_mode == PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
        elif pooling_mode == PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
        elif pooling_mode == PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
        else:
            # This proves that we have exhaustively checked all PoolingModes
            raise RuntimeError("Unknown PoolingMode!")

        # stochastic rounding only implemented for rowwise
        assume(not stochastic_rounding or row_wise)
        # only row-wise supports caching
        assume(row_wise or not use_cache)

        E = int(10**log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]

        if not mixed_B:
            Bs = [B] * T
        else:
            low = max(int(0.25 * B), 1)
            high = int(B)
            if low == high:
                Bs = [B] * T
            else:
                Bs = [np.random.randint(low=low, high=high) for _ in range(T)]

        compute_device = ComputeDevice.CUDA
        if use_cpu:
            managed = [EmbeddingLocation.HOST] * T
            compute_device = ComputeDevice.CPU
        elif TEST_WITH_ROCM:
            # ROCm managed memory allocation is under development
            managed = [EmbeddingLocation.DEVICE] * T
        elif use_cache:
            managed = [EmbeddingLocation.MANAGED_CACHING] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        EmbeddingLocation.DEVICE if d < average_D else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        EmbeddingLocation.DEVICE,
                        EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        if do_pooling:
            bs = [
                to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]
        else:
            bs = [
                to_device(torch.nn.Embedding(E, D, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]

        if weights_precision == SparseType.FP16:
            bs = [b.half() for b in bs]

        feature_table_map = list(range(T))
        # autograd with shared embedding only works for exact
        table_to_replicate = T // 2
        # pyre-fixme[6]: For 2nd param expected `Embedding` but got
        #  `Union[Embedding, EmbeddingBag]`.
        bs.insert(table_to_replicate, bs[table_to_replicate])
        feature_table_map.insert(table_to_replicate, table_to_replicate)

        num_features = len(feature_table_map)
        if not mixed_B:
            Bs = [B] * num_features
            Bs_rank_feature = [[0]]
        else:
            Bs_rank_feature, Bs = gen_mixed_B_batch_sizes(B, num_features)

        xs = [
            to_device(
                torch.from_numpy(
                    np.random.choice(range(Es[t]), size=(b, L), replace=True).astype(
                        np.int64
                    )
                ),
                use_cpu,
            )
            for t, b in zip(feature_table_map, Bs)
        ]
        xws = [to_device(torch.randn(size=(b, L)), use_cpu) for b in Bs]
        xws_acc_type = copy.deepcopy(xws)

        if weights_precision == SparseType.FP16 and not use_cpu:
            xws = [xw.half() for xw in xws]

        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, xs)
            ]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        gos = [torch.randn_like(f) for f in fs]
        [f.backward(go) for (f, go) in zip(fs, gos)]
        # do SGD update
        lr = 0.5
        eps = 0.2

        optimizer = (
            OptimType.EXACT_ROWWISE_ADAGRAD if row_wise else OptimType.EXACT_ADAGRAD
        )
        cc = emb_op(
            embedding_specs=[
                (E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)
            ],
            feature_table_map=feature_table_map,
            optimizer=optimizer,
            learning_rate=lr,
            eps=eps,
            weights_precision=weights_precision,
            stochastic_rounding=stochastic_rounding,
            pooling_mode=pooling_mode,
            output_dtype=output_dtype,
        )

        del bs[table_to_replicate]
        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.contiguous().flatten() for x in xs], dim=0)
        xw = torch.cat([xw.contiguous().flatten() for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(
            x, L, sum(Bs), use_cpu=use_cpu
        )

        batch_size_per_feature_per_rank = Bs_rank_feature if mixed_B else None

        fc2 = (
            cc(
                indices,
                offsets,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
            if not weighted
            else cc(
                indices,
                offsets,
                to_device(xw.contiguous().view(-1), use_cpu),
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
        )
        if do_pooling:
            if mixed_B:
                goc = format_ref_tensors_in_mixed_B_layout(gos, Bs_rank_feature)
            else:
                goc = torch.cat([go.view(B, -1) for go in gos], dim=1)
        else:
            goc = torch.cat(gos, dim=0)
        fc2.backward(goc)
        cc.flush()
        split_optimizer_states = cc.split_optimizer_states()
        assert len(split_optimizer_states) == T

        get_optimizer_states = None
        if row_wise:
            # get_optimizer_state should/must be implemented for rowwise
            get_optimizer_states = cc.get_optimizer_state()
            assert len(get_optimizer_states) == T

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        for t in range(T):
            expected_keys = {"sum"}
            if row_wise and weight_decay_mode == WeightDecayMode.COUNTER:
                (m1, c1, c2) = split_optimizer_states[t]
                expected_keys.update(
                    [
                        "prev_iter",
                        "row_counter",
                    ]
                )
            else:
                (m1,) = split_optimizer_states[t]
            if get_optimizer_states is not None:
                optimizer_states_dict = get_optimizer_states[t]
                assert set(optimizer_states_dict.keys()) == expected_keys
            # pyre-fixme[16]: `Optional` has no attribute `float`.
            ref_optimizer_state = bs[t].weight.grad.float().cpu().to_dense().pow(2)
            torch.testing.assert_close(
                m1.float().cpu(),
                ref_optimizer_state.mean(dim=1) if row_wise else ref_optimizer_state,
                atol=tolerance,
                rtol=tolerance,
            )
        for t in range(T):
            # optimizer_state = squares (no row-wise) or sum squares (row-wise)
            if row_wise and weight_decay_mode == WeightDecayMode.COUNTER:
                (m1, c1, c2) = split_optimizer_states[t]
            else:
                (m1,) = split_optimizer_states[t]
            torch.testing.assert_close(
                cc.split_embedding_weights()[t].float().cpu(),
                torch.addcdiv(
                    bs[t].weight.float().cpu(),
                    value=-lr,
                    tensor1=bs[t].weight.grad.float().cpu().to_dense(),
                    tensor2=m1.float()
                    .sqrt_()
                    .add_(eps)
                    .view(Es[t], 1 if row_wise else Ds[t])
                    .cpu(),
                ),
                atol=tolerance,
                rtol=tolerance,
            )
        if use_cpu:
            D_gradcheck = (D_gradcheck + 15) // 16 * 4
        else:
            D_gradcheck = D_gradcheck * 4
        cc = emb_op(
            embedding_specs=[
                (E, D_gradcheck, M, compute_device) for (E, M) in zip(Es, managed)
            ],
            feature_table_map=feature_table_map,
            optimizer=optimizer,
            learning_rate=0.0,
            eps=eps,
            weights_precision=weights_precision,
            stochastic_rounding=stochastic_rounding,
            # NOTE: only SUM pooling can work with per_sample_weights!
            pooling_mode=PoolingMode.SUM,
            output_dtype=output_dtype,
        )
        per_sample_weights = to_device(xw.contiguous().view(-1), use_cpu)
        if use_cpu:
            # NOTE: GPU version of SplitTableBatchedEmbeddingBagsCodegen doesn't support double.
            cc = cc.double()
            per_sample_weights = per_sample_weights.double()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        for param in cc.parameters():
            param.requires_grad = False
        torch.autograd.gradcheck(
            cc,
            (
                indices,
                offsets,
                per_sample_weights,
                None,
                batch_size_per_feature_per_rank,
            ),
        )

        per_sample_weights = to_device(xw.contiguous().view(-1), use_cpu)
        if use_cpu:
            per_sample_weights = per_sample_weights.double()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        for param in cc.parameters():
            param.requires_grad = False
        y = cc(
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )
        y.sum().backward()
        # pyre-fixme[16]: `Optional` has no attribute `clone`.
        indice_weight_grad_all = per_sample_weights.grad.clone().cpu()
        T_ = len(xws)
        feature_requires_grad = to_device(
            torch.tensor(np.random.choice([0, 1], replace=True, size=(T_,))).int(),
            use_cpu,
        )
        per_sample_weights = per_sample_weights.detach().clone()
        per_sample_weights.requires_grad = True
        y = cc(
            indices,
            offsets,
            per_sample_weights,
            feature_requires_grad=feature_requires_grad,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )
        y.sum().backward()
        indice_weight_grad_mask = per_sample_weights.grad.clone().cpu()
        torch.cuda.synchronize()

        acc_B = 0
        for t in range(T_):
            B = Bs[t]
            table_indice_weight_grad_mask = indice_weight_grad_mask[
                acc_B : acc_B + B * L
            ]
            table_indice_weight_grad_all = indice_weight_grad_all[acc_B : acc_B + B * L]
            acc_B += B * L
            if feature_requires_grad[t]:
                torch.testing.assert_close(
                    table_indice_weight_grad_mask,
                    table_indice_weight_grad_all,
                )
            else:
                torch.testing.assert_close(
                    table_indice_weight_grad_mask,
                    torch.zeros_like(table_indice_weight_grad_mask),
                )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        D_gradcheck=st.integers(min_value=1, max_value=2),
        weights_precision=st.just(SparseType.FP16),
        stochastic_rounding=st.booleans(),
        weighted=st.booleans(),
        row_wise=st.booleans(),
        mixed=st.booleans(),
        mixed_B=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_adagrad_fp16_pmSUM(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        D_gradcheck: int,
        weights_precision: SparseType,
        stochastic_rounding: bool,
        weighted: bool,
        row_wise: bool,
        mixed: bool,
        mixed_B: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        use_cpu: bool,
        output_dtype: SparseType,
    ) -> None:
        # VBE is supported in rowwise_adagrad only
        if not row_wise:
            mixed_B = False
        self.execute_backward_adagrad_(
            T,
            D,
            B,
            log_E,
            L,
            D_gradcheck,
            weights_precision,
            stochastic_rounding,
            weighted,
            row_wise,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            PoolingMode.SUM,
            use_cpu,
            output_dtype,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        D_gradcheck=st.integers(min_value=1, max_value=2),
        weights_precision=st.just(SparseType.FP16),
        stochastic_rounding=st.booleans(),
        weighted=st.booleans(),
        row_wise=st.booleans(),
        mixed=st.booleans(),
        mixed_B=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_adagrad_fp16_pmMEAN(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        D_gradcheck: int,
        weights_precision: SparseType,
        stochastic_rounding: bool,
        weighted: bool,
        row_wise: bool,
        mixed: bool,
        mixed_B: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        use_cpu: bool,
        output_dtype: SparseType,
    ) -> None:
        # VBE is supported in rowwise_adagrad only
        if not row_wise:
            mixed_B = False
        self.execute_backward_adagrad_(
            T,
            D,
            B,
            log_E,
            L,
            D_gradcheck,
            weights_precision,
            stochastic_rounding,
            weighted,
            row_wise,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            PoolingMode.MEAN,
            use_cpu,
            output_dtype,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        D_gradcheck=st.integers(min_value=1, max_value=2),
        weights_precision=st.just(SparseType.FP16),
        stochastic_rounding=st.booleans(),
        weighted=st.booleans(),
        row_wise=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_adagrad_fp16_pmNONE(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        D_gradcheck: int,
        weights_precision: SparseType,
        stochastic_rounding: bool,
        weighted: bool,
        row_wise: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        use_cpu: bool,
        output_dtype: SparseType,
    ) -> None:
        self.execute_backward_adagrad_(
            T,
            D,
            B,
            log_E,
            L,
            D_gradcheck,
            weights_precision,
            stochastic_rounding,
            weighted,
            row_wise,
            mixed,
            False,  # mixed_B
            use_cache,
            cache_algorithm,
            PoolingMode.NONE,
            use_cpu,
            output_dtype,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        D_gradcheck=st.integers(min_value=1, max_value=2),
        weights_precision=st.just(SparseType.FP32),
        stochastic_rounding=st.booleans(),
        weighted=st.booleans(),
        row_wise=st.booleans(),
        mixed=st.booleans(),
        mixed_B=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_adagrad_fp32_pmSUM(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        D_gradcheck: int,
        weights_precision: SparseType,
        stochastic_rounding: bool,
        weighted: bool,
        row_wise: bool,
        mixed: bool,
        mixed_B: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        use_cpu: bool,
        output_dtype: SparseType,
    ) -> None:
        # VBE is supported in rowwise_adagrad only
        if not row_wise:
            mixed_B = False
        self.execute_backward_adagrad_(
            T,
            D,
            B,
            log_E,
            L,
            D_gradcheck,
            weights_precision,
            stochastic_rounding,
            weighted,
            row_wise,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            PoolingMode.SUM,
            use_cpu,
            output_dtype,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        D_gradcheck=st.integers(min_value=1, max_value=2),
        weights_precision=st.just(SparseType.FP32),
        stochastic_rounding=st.booleans(),
        weighted=st.booleans(),
        row_wise=st.booleans(),
        mixed=st.booleans(),
        mixed_B=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_adagrad_fp32_pmMEAN(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        D_gradcheck: int,
        weights_precision: SparseType,
        stochastic_rounding: bool,
        weighted: bool,
        row_wise: bool,
        mixed: bool,
        mixed_B: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        use_cpu: bool,
        output_dtype: SparseType,
    ) -> None:
        # VBE is supported in rowwise_adagrad only
        if not row_wise:
            mixed_B = False
        self.execute_backward_adagrad_(
            T,
            D,
            B,
            log_E,
            L,
            D_gradcheck,
            weights_precision,
            stochastic_rounding,
            weighted,
            row_wise,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            PoolingMode.MEAN,
            use_cpu,
            output_dtype,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        D_gradcheck=st.integers(min_value=1, max_value=2),
        weights_precision=st.just(SparseType.FP32),
        stochastic_rounding=st.booleans(),
        weighted=st.booleans(),
        row_wise=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_adagrad_fp32_pmNONE(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        D_gradcheck: int,
        weights_precision: SparseType,
        stochastic_rounding: bool,
        weighted: bool,
        row_wise: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        use_cpu: bool,
        output_dtype: SparseType,
    ) -> None:
        self.execute_backward_adagrad_(
            T,
            D,
            B,
            log_E,
            L,
            D_gradcheck,
            weights_precision,
            stochastic_rounding,
            weighted,
            row_wise,
            mixed,
            False,  # mixed_B
            use_cache,
            cache_algorithm,
            PoolingMode.NONE,
            use_cpu,
            output_dtype,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        mixed=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_pipeline(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
        cache_algorithm: CacheAlgorithm,
    ) -> None:
        iters = 3
        E = int(10**log_E)
        D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        managed = [EmbeddingLocation.MANAGED_CACHING] * T
        if mixed:
            average_D = sum(Ds) // T
            for t, d in enumerate(Ds):
                managed[t] = EmbeddingLocation.DEVICE if d < average_D else managed[t]
        cc_ref = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    D,
                    EmbeddingLocation.DEVICE,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
        )
        cc = SplitTableBatchedEmbeddingBagsCodegen(
            [(E, D, M, ComputeDevice.CUDA) for (E, D, M) in zip(Es, Ds, managed)],
            cache_algorithm=cache_algorithm,
        )
        for t in range(T):
            self.assertEqual(
                cc.split_embedding_weights()[t].size(),
                cc_ref.split_embedding_weights()[t].size(),
            )
            cc.split_embedding_weights()[t].data.copy_(
                cc_ref.split_embedding_weights()[t]
            )

        requests = generate_requests(iters, B, T, L, min(Es), reuse=0.1)
        grad_output = torch.randn(B, sum(Ds)).cuda()

        for indices, offsets, _ in requests:
            output = cc(indices, offsets)
            output_ref = cc_ref(indices, offsets)
            torch.testing.assert_close(output, output_ref)
            output.backward(grad_output)
            output_ref.backward(grad_output)
        cc.flush()
        for t in range(T):
            torch.testing.assert_close(
                cc.split_embedding_weights()[t], cc_ref.split_embedding_weights()[t]
            )

    def execute_backward_optimizers_(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        mixed_B: bool,
        optimizer: OptimType,
        long_segments: bool,
        pooling_mode: PoolingMode,
        use_cpu: bool,
        weight_decay_mode: WeightDecayMode = WeightDecayMode.L2,
        uvm_non_rowwise_momentum: bool = False,
    ) -> None:
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        assume(not use_cpu or T * B * L * D <= 2048)
        assume(
            not use_cpu
            or optimizer
            in [
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_SGD,
            ]
        )

        assume(pooling_mode == PoolingMode.SUM or not weighted)
        # No bag ops only work on GPUs, no mixed, no weighted
        assume(not use_cpu or pooling_mode != PoolingMode.NONE)
        assume(not mixed or pooling_mode != PoolingMode.NONE)
        assume(not weighted or pooling_mode != PoolingMode.NONE)
        assume(not mixed_B or (not use_cpu and pooling_mode != PoolingMode.NONE))

        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        if pooling_mode == PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
        elif pooling_mode == PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
        elif pooling_mode == PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
        else:
            # This proves that we have exhaustively checked all PoolingModes
            raise RuntimeError("Unknown PoolingMode!")

        E = int(10**log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]

        if not mixed_B:
            Bs = [B] * T
            Bs_rank_feature = [[0]]
        else:
            Bs_rank_feature, Bs = gen_mixed_B_batch_sizes(B, T)

        compute_device = ComputeDevice.CUDA
        if use_cpu:
            managed = [EmbeddingLocation.HOST] * T
            compute_device = ComputeDevice.CPU
        elif TEST_WITH_ROCM:
            # ROCm managed memory allocation is under development
            managed = [EmbeddingLocation.DEVICE] * T
        else:
            managed = [
                np.random.choice(
                    [
                        EmbeddingLocation.DEVICE,
                        EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        if do_pooling:
            bs = [
                to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]
        else:
            bs = [
                to_device(torch.nn.Embedding(E, D, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]

        xs = [
            to_device(
                torch.from_numpy(
                    np.random.choice(range(e), size=(b, L), replace=True).astype(
                        np.int64
                    )
                ),
                use_cpu,
            )
            for (e, b) in zip(Es, Bs)
        ]
        if long_segments and L > 0:
            for x, e in zip(xs, Es):
                x[:, 0] = np.random.randint(low=0, high=e)

        xws = [to_device(torch.randn(size=(b, L)), use_cpu) for b in Bs]
        xws_acc_type = copy.deepcopy(xws)

        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, xs)
            ]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        gos = [torch.randn_like(f) for f in fs]
        [f.backward(go) for (f, go) in zip(fs, gos)]
        # do SGD update

        optimizer_kwargs = {"learning_rate": 0.5}
        (lr, eps, beta1, beta2, weight_decay, momentum, eta) = (
            0.5,
            1e-4,
            0.9,
            0.99,
            0.01,
            0.9,
            0.01,
        )
        counter_based_regularization: CounterBasedRegularizationDefinition

        if optimizer == OptimType.EXACT_ADAGRAD:
            optimizer_kwargs["eps"] = eps

        if optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
            optimizer_kwargs["eps"] = eps
            optimizer_kwargs["weight_decay"] = weight_decay
            optimizer_kwargs["weight_decay_mode"] = weight_decay_mode
            if weight_decay_mode == WeightDecayMode.COUNTER:
                counter_based_regularization = CounterBasedRegularizationDefinition(
                    counter_weight_decay_mode=CounterWeightDecayMode.DECOUPLE,
                    counter_halflife=20000,
                    adjustment_iter=24000,
                    adjustment_ub=0.1,
                    learning_rate_mode=LearningRateMode.TAIL_ID_LR_DECREASE,
                    grad_sum_decay=GradSumDecay.NO_DECAY,
                    tail_id_threshold=TailIdThreshold(val=1000, is_ratio=False),
                )

                optimizer_kwargs[
                    "counter_based_regularization"
                    # pyre-fixme[6]: Expected `float` for 2nd param but got `CounterBasedRegularizationDefinition`.
                ] = counter_based_regularization

        if optimizer == OptimType.EXACT_ROWWISE_WEIGHTED_ADAGRAD:
            optimizer_kwargs["eps"] = eps
            optimizer_kwargs["weight_decay"] = weight_decay

        if optimizer in (OptimType.PARTIAL_ROWWISE_ADAM, OptimType.ADAM):
            optimizer_kwargs["eps"] = eps
            optimizer_kwargs["beta1"] = beta1
            optimizer_kwargs["beta2"] = beta2
            optimizer_kwargs["weight_decay"] = weight_decay

        if optimizer in (OptimType.PARTIAL_ROWWISE_LAMB, OptimType.LAMB):
            optimizer_kwargs["eps"] = eps
            optimizer_kwargs["beta1"] = beta1
            optimizer_kwargs["beta2"] = beta2
            optimizer_kwargs["weight_decay"] = weight_decay

        if optimizer == OptimType.LARS_SGD:
            optimizer_kwargs["weight_decay"] = weight_decay
            optimizer_kwargs["momentum"] = momentum
            optimizer_kwargs["eta"] = eta

        cc = emb_op(
            embedding_specs=[
                (E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)
            ],
            optimizer=optimizer,
            pooling_mode=pooling_mode,
            uvm_non_rowwise_momentum=uvm_non_rowwise_momentum,
            # pyre-fixme[6]: Expected `CacheAlgorithm` for 5th param but got `float`.
            **optimizer_kwargs,
        )

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.contiguous().flatten() for x in xs], dim=0)
        xw = torch.cat([xw.contiguous().flatten() for xw in xws_acc_type], dim=0)

        batch_size_per_feature_per_rank = Bs_rank_feature if mixed_B else None

        (indices, offsets) = get_table_batched_offsets_from_dense(
            x, L, sum(Bs), use_cpu=use_cpu
        )
        fc2 = (
            cc(
                indices,
                offsets,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
            if not weighted
            else cc(
                indices,
                offsets,
                to_device(xw.contiguous().view(-1), use_cpu),
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
        )
        if do_pooling:
            if mixed_B:
                goc = format_ref_tensors_in_mixed_B_layout(gos, Bs_rank_feature)
            else:
                goc = torch.cat([go.view(B, -1) for go in gos], dim=1)
        else:
            goc = torch.cat(gos, dim=0)
        fc2.backward(goc)
        cc.flush()

        split_optimizer_states = cc.split_optimizer_states()

        self.assertEqual(len(split_optimizer_states), T)
        split_weights = cc.split_embedding_weights()

        get_optimizer_states = None

        try:
            get_optimizer_states = cc.get_optimizer_state()
            assert len(get_optimizer_states) == T
        except NotImplementedError:
            assert optimizer not in (
                OptimType.ADAM,
                OptimType.PARTIAL_ROWWISE_ADAM,
                OptimType.LAMB,
                OptimType.PARTIAL_ROWWISE_LAMB,
                OptimType.EXACT_SGD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_ROWWISE_WEIGHTED_ADAGRAD,
                OptimType.EXACT_ADAGRAD,
            )

        if optimizer in (OptimType.EXACT_ROWWISE_ADAGRAD, OptimType.EXACT_ADAGRAD):
            rowwise = optimizer == OptimType.EXACT_ROWWISE_ADAGRAD
            for t in range(T):
                row_counter: Optional[torch.Tensor] = None
                freq: Optional[torch.Tensor] = None
                iter_: int = -1

                if rowwise and weight_decay_mode == WeightDecayMode.COUNTER:
                    (m1, prev_iter, row_counter) = split_optimizer_states[t]
                else:
                    (m1,) = split_optimizer_states[t]
                # to_dense in GPU is non-deterministic due to atmomics used in
                # coalescing and floating point non-associativity.
                # pyre-fixme[16]: `Optional` has no attribute `cpu`.
                dense_cpu_grad = bs[t].weight.grad.cpu().to_dense()
                if rowwise and not use_cpu:
                    # We need to skip when using cpu because use_fbgemm (https://fburl.com/code/12131iub)
                    # is true and the template code (https://fburl.com/code/1kctlup3) is not executed.
                    if weight_decay_mode == WeightDecayMode.L2:
                        dense_cpu_grad += weight_decay * bs[t].weight.cpu()
                    elif weight_decay_mode == WeightDecayMode.COUNTER:
                        iter_ = int(cc.iter.item())
                        (
                            dense_cpu_grad,
                            row_counter,
                            freq,
                        ) = self.get_grad_from_counter_adagrad(
                            dense_cpu_grad,
                            bs[t].weight.cpu(),
                            counter_based_regularization,
                            row_counter.cpu(),
                            prev_iter.cpu(),
                            iter_,
                            weight_decay,
                        )

                m1_ref = (
                    dense_cpu_grad.pow(2)
                    if not rowwise
                    else dense_cpu_grad.pow(2).mean(dim=1)
                )
                torch.testing.assert_close(
                    m1.float().index_select(dim=0, index=xs[t].view(-1)).cpu(),
                    m1_ref.float().index_select(dim=0, index=xs[t].view(-1).cpu()),
                    atol=1.0e-4,
                    rtol=1.0e-4,
                )
                weights_new = split_weights[t]
                denom = (
                    torch.sqrt(
                        m1_ref if not rowwise else m1_ref.view(m1_ref.numel(), 1)
                    )
                    + eps
                )
                if rowwise and not use_cpu:
                    if weight_decay_mode == WeightDecayMode.DECOUPLE:
                        weights_ref = bs[t].weight.cpu() - lr * (
                            dense_cpu_grad / denom + weight_decay * bs[t].weight.cpu()
                        )
                    elif weight_decay_mode == WeightDecayMode.L2:
                        # pyre-fixme[58]: `/` is not supported for operand types `float`
                        #  and `Tensor`.
                        weights_ref = bs[t].weight.cpu() - lr * dense_cpu_grad / denom
                    elif weight_decay_mode == WeightDecayMode.COUNTER:
                        max_counter = cc.max_counter.item()
                        weights_ref = self.get_wts_from_counter_adagrad(
                            dense_cpu_grad,
                            bs[t].weight.cpu(),
                            denom,
                            counter_based_regularization,
                            row_counter,
                            # pyre-fixme[6]: Expected `Tensor` for 6th param but got `Optional[Tensor]`
                            freq,
                            max_counter,
                            iter_,
                            eps,
                            lr,
                            weight_decay,
                        )
                else:
                    # pyre-fixme[58]: `/` is not supported for operand types `float`
                    #  and `Tensor`.
                    weights_ref = bs[t].weight.cpu() - lr * dense_cpu_grad / denom
                # TODO: why is tolerance off here?
                torch.testing.assert_close(
                    weights_new.index_select(dim=0, index=xs[t].view(-1)).cpu(),
                    weights_ref.index_select(dim=0, index=xs[t].view(-1).cpu()),
                    atol=1.0e-2,
                    rtol=1.0e-2,
                )

                optimizer_states_dict = get_optimizer_states[t]
                expected_keys = {"sum"}
                if rowwise and weight_decay_mode == WeightDecayMode.COUNTER:
                    expected_keys.update(["prev_iter", "row_counter"])
                assert set(optimizer_states_dict.keys()) == expected_keys

        if optimizer == OptimType.EXACT_ROWWISE_WEIGHTED_ADAGRAD:
            for t in range(T):
                (m1,) = split_optimizer_states[t]
                # to_dense in GPU is non-deterministic due to atmomics used in
                # coalescing and floating point non-associativity.
                dense_cpu_grad = bs[t].weight.grad.cpu().to_dense()
                dense_cpu_grad += weight_decay * bs[t].weight.cpu()
                iter_ = cc.iter.item()
                lambda_ = (iter_ + 1) ** 0.5
                m1_ref = dense_cpu_grad.pow(2).mean(dim=1)
                m1_ref *= lambda_
                torch.testing.assert_close(
                    m1.float().index_select(dim=0, index=xs[t].view(-1)).cpu(),
                    m1_ref.float().index_select(dim=0, index=xs[t].view(-1).cpu()),
                    atol=1.0e-4,
                    rtol=1.0e-4,
                )
                weights_new = split_weights[t]
                weights_ref = bs[t].weight.cpu() - lr * lambda_ * dense_cpu_grad / (
                    # pyre-fixme[58]: `/` is not supported for operand types `float`
                    #  and `Tensor`.
                    torch.pow(m1_ref.view(m1_ref.numel(), 1), 1.0 / 3)
                    + eps
                )
                torch.testing.assert_close(
                    weights_new.index_select(dim=0, index=xs[t].view(-1)).cpu(),
                    weights_ref.index_select(dim=0, index=xs[t].view(-1).cpu()),
                    atol=1.0e-4,
                    rtol=1.0e-4,
                )

                if get_optimizer_states is not None:
                    optimizer_states_dict = get_optimizer_states[t]
                    assert set(optimizer_states_dict.keys()) == {"sum"}

        if optimizer in (OptimType.PARTIAL_ROWWISE_ADAM, OptimType.ADAM):
            rowwise = optimizer == OptimType.PARTIAL_ROWWISE_ADAM
            for t in range(T):
                (m1, m2) = split_optimizer_states[t]
                dense_cpu_grad = bs[t].weight.grad.cpu().to_dense()
                m2_ref = (
                    dense_cpu_grad.pow(2)
                    if not rowwise
                    else dense_cpu_grad.pow(2).mean(dim=1)
                ) * (1.0 - beta2)
                torch.testing.assert_close(m2.cpu(), m2_ref, atol=1.0e-4, rtol=1.0e-4)
                m1_ref = dense_cpu_grad * (1.0 - beta1)
                torch.testing.assert_close(m1.cpu(), m1_ref, atol=1.0e-4, rtol=1.0e-4)
                iter_ = cc.iter.item()
                v_hat_t = m2_ref / (1 - beta2**iter_)
                v_hat_t = v_hat_t if not rowwise else v_hat_t.view(v_hat_t.numel(), 1)
                m_hat_t = m1_ref / (1 - beta1**iter_)
                weights_new = split_weights[t]
                weights_ref = (
                    torch.addcdiv(
                        bs[t].weight.cpu(),
                        value=-lr,
                        tensor1=m_hat_t,
                        tensor2=v_hat_t.sqrt_().add_(eps),
                    )
                    - lr * weight_decay * bs[t].weight.cpu()
                )
                torch.testing.assert_close(
                    weights_new.index_select(dim=0, index=xs[t].view(-1)).cpu(),
                    weights_ref.index_select(dim=0, index=xs[t].view(-1).cpu()),
                    atol=1.0e-3,
                    rtol=1.0e-3,
                )

                if get_optimizer_states is not None:
                    optimizer_states_dict = get_optimizer_states[t]
                    assert set(optimizer_states_dict.keys()) == {
                        "exp_avg",
                        "exp_avg_sq",
                    }

        if optimizer in (OptimType.PARTIAL_ROWWISE_LAMB, OptimType.LAMB):
            rowwise = optimizer == OptimType.PARTIAL_ROWWISE_LAMB
            for t in range(T):
                (m1, m2) = split_optimizer_states[t]
                dense_cpu_grad = bs[t].weight.grad.cpu().to_dense()
                m2_ref = (
                    dense_cpu_grad.pow(2)
                    if not rowwise
                    else dense_cpu_grad.pow(2).mean(dim=1)
                ) * (1.0 - beta2)
                torch.testing.assert_close(m2.cpu(), m2_ref, atol=1.0e-4, rtol=1.0e-4)
                m1_ref = dense_cpu_grad * (1.0 - beta1)
                torch.testing.assert_close(m1.cpu(), m1_ref, atol=1.0e-4, rtol=1.0e-4)
                iter_ = cc.iter.item()
                v_hat_t = m2_ref / (1 - beta2**iter_)
                v_hat_t = v_hat_t if not rowwise else v_hat_t.view(v_hat_t.numel(), 1)
                m_hat_t = m1_ref / (1 - beta1**iter_)
                rtw = (m_hat_t / (torch.sqrt(v_hat_t) + eps)) + weight_decay * bs[
                    t
                ].weight.cpu()
                true_ratio = torch.linalg.norm(bs[t].weight, dim=1, ord=2).view(
                    m1.shape[0], 1
                ).cpu() / torch.linalg.norm(rtw, dim=1, ord=2).view(m1.shape[0], 1)
                weights_new = split_weights[t]
                weights_ref = bs[t].weight.cpu() - lr * true_ratio * rtw
                torch.testing.assert_close(
                    weights_new.index_select(dim=0, index=xs[t].view(-1)).cpu(),
                    weights_ref.index_select(dim=0, index=xs[t].view(-1).cpu()),
                    atol=1.0e-3,
                    rtol=1.0e-3,
                )
                if get_optimizer_states is not None:
                    optimizer_states_dict = get_optimizer_states[t]
                    assert set(optimizer_states_dict.keys()) == {
                        "exp_avg",
                        "exp_avg_sq",
                    }

        if optimizer == OptimType.LARS_SGD:
            for t in range(T):
                (m1,) = split_optimizer_states[t]
                weight_norm = (
                    torch.linalg.norm(bs[t].weight, dim=1, ord=2)
                    .view(m1.shape[0], 1)
                    .cpu()
                )
                dense_cpu_grad = bs[t].weight.grad.cpu().to_dense()
                grad_norm = torch.linalg.norm(dense_cpu_grad, dim=1, ord=2).view(
                    m1.shape[0], 1
                )
                adjusted_lr = (
                    lr * eta * weight_norm / (grad_norm + weight_decay * weight_norm)
                )
                m1_ref = adjusted_lr * (
                    dense_cpu_grad + weight_decay * bs[t].weight.cpu()
                )

                torch.testing.assert_close(
                    m1.index_select(dim=0, index=xs[t].view(-1)).cpu(),
                    # pyre-fixme[16]: `float` has no attribute `index_select`.
                    m1_ref.index_select(dim=0, index=xs[t].view(-1).cpu()),
                    atol=1.0e-4,
                    rtol=1.0e-4,
                )
                weights_new = split_weights[t]
                weights_ref = bs[t].weight.cpu() - m1_ref
                torch.testing.assert_close(
                    weights_new.index_select(dim=0, index=xs[t].view(-1)).cpu(),
                    weights_ref.index_select(dim=0, index=xs[t].view(-1).cpu()),
                    atol=1.0e-4,
                    rtol=1.0e-4,
                )

    def get_grad_from_counter_adagrad(
        self,
        dense_cpu_grad: torch.Tensor,
        weights: torch.Tensor,
        counter_based_regularization: CounterBasedRegularizationDefinition,
        row_counter: torch.Tensor,
        prev_iter: torch.Tensor,
        iter_: int,
        weight_decay: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_counter = row_counter.view(row_counter.numel(), 1)
        prev_iter = prev_iter.view(prev_iter.numel(), 1)
        freq = torch.ones_like(row_counter)
        counter_weight_decay_mode = (
            counter_based_regularization.counter_weight_decay_mode
        )
        counter_halflife = counter_based_regularization.counter_halflife
        l2_wd = 1.0 if counter_weight_decay_mode == CounterWeightDecayMode.L2 else 0.0

        if counter_halflife > 0:
            counter_log_rho = math.log(2.0) / counter_halflife
            # if id occurs multiple times in a batch, iter_delta=1
            iter_delta = torch.where(prev_iter == 0.0, 1.0, iter_ * 1.0 - prev_iter)
            prev_iter = iter_ * torch.ones_like(prev_iter)
            row_counter = 1.0 + torch.exp(-iter_delta * counter_log_rho) * row_counter
            freq = torch.tensor([counter_halflife]) / row_counter

        dense_cpu_grad += l2_wd * freq * weight_decay * weights
        return dense_cpu_grad, row_counter, freq

    def get_wts_from_counter_adagrad(
        self,
        dense_cpu_grad: torch.Tensor,
        weights: torch.Tensor,
        denom: torch.Tensor,
        counter_based_regularization: CounterBasedRegularizationDefinition,
        row_counter: torch.Tensor,
        freq: torch.Tensor,
        max_counter: float,
        iter_: int,
        eps: float,
        learning_rate: float,
        weight_decay: float,
    ) -> torch.Tensor:
        counter_weight_decay_mode = (
            counter_based_regularization.counter_weight_decay_mode
        )
        counter_halflife = counter_based_regularization.counter_halflife
        tail_id_threshold_val = counter_based_regularization.tail_id_threshold.val
        if counter_based_regularization.tail_id_threshold.is_ratio:
            tail_id_threshold_val = math.floor(tail_id_threshold_val * max_counter)
        learning_rate_mode = counter_based_regularization.learning_rate_mode
        adjustment_iter = counter_based_regularization.adjustment_iter
        adjustment_ub = counter_based_regularization.adjustment_ub

        multiplier = torch.tensor([learning_rate]) / denom
        adjusted_multiplier = multiplier
        exp_reg_correction = torch.ones_like(row_counter)

        if counter_halflife > 0:
            if adjustment_iter <= 0 or (
                adjustment_iter > 0 and iter_ > adjustment_iter
            ):
                if learning_rate_mode == LearningRateMode.TAIL_ID_LR_INCREASE:
                    adjusted_multiplier = torch.where(
                        row_counter > tail_id_threshold_val,
                        multiplier
                        * torch.maximum(
                            torch.minimum(
                                torch.pow(
                                    torch.tensor([max_counter]) / (row_counter + 1.0),
                                    adjustment_ub,
                                ),
                                torch.Tensor([10.0]),
                            ),
                            torch.Tensor([1.0]),
                        ),
                        multiplier,
                    )
                elif learning_rate_mode == LearningRateMode.TAIL_ID_LR_DECREASE:
                    adjusted_multiplier = torch.where(
                        row_counter > tail_id_threshold_val,
                        multiplier
                        * torch.minimum(
                            torch.maximum(
                                torch.pow(
                                    (row_counter + 1.0) / max_counter,
                                    adjustment_ub,
                                ),
                                torch.Tensor([0.1]),
                            ),
                            torch.Tensor([1.0]),
                        ),
                        multiplier,
                    )
                elif learning_rate_mode == LearningRateMode.COUNTER_SGD:
                    adjusted_multiplier = torch.where(
                        row_counter > tail_id_threshold_val,
                        torch.Tensor([learning_rate])
                        / (torch.sqrt(adjustment_ub * row_counter) + eps),
                        multiplier,
                    )

                if counter_weight_decay_mode == CounterWeightDecayMode.DECOUPLE:
                    exp_reg_correction = 1.0 - freq * weight_decay * learning_rate
                elif counter_weight_decay_mode == CounterWeightDecayMode.L2:
                    exp_reg_correction = 1.0 - freq * weight_decay * multiplier

        weights = exp_reg_correction * weights - adjusted_multiplier * dense_cpu_grad
        return weights

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        optimizer=st.sampled_from(
            [
                OptimType.ADAM,
                OptimType.PARTIAL_ROWWISE_ADAM,
            ]
        ),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        uvm_non_rowwise_momentum=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_optimizers_adam(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        optimizer: OptimType,
        long_segments: bool,
        pooling_mode: PoolingMode,
        use_cpu: bool,
        uvm_non_rowwise_momentum: bool,
    ) -> None:
        self.execute_backward_optimizers_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            False,  # mixed_B
            optimizer,
            long_segments,
            pooling_mode,
            use_cpu,
            uvm_non_rowwise_momentum=uvm_non_rowwise_momentum,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=2, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        mixed_B=st.booleans(),
        optimizer=st.sampled_from(
            [
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_ROWWISE_WEIGHTED_ADAGRAD,
            ]
        ),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        weight_decay_mode=st.sampled_from(
            [
                WeightDecayMode.L2,
                WeightDecayMode.DECOUPLE,
                # temporarily disabled due to a test error to unblock release
                # will fix in a follow-up diff
                # WeightDecayMode.COUNTER,
            ]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_optimizers_adagrad(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        mixed_B: bool,
        optimizer: OptimType,
        long_segments: bool,
        pooling_mode: PoolingMode,
        use_cpu: bool,
        weight_decay_mode: WeightDecayMode,
    ) -> None:
        if (
            pooling_mode == PoolingMode.NONE
            or optimizer != OptimType.EXACT_ROWWISE_ADAGRAD
        ):
            mixed_B = False
        self.execute_backward_optimizers_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            mixed_B,
            optimizer,
            long_segments,
            pooling_mode,
            use_cpu,
            weight_decay_mode,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        optimizer=st.sampled_from(
            [
                OptimType.LAMB,
                OptimType.PARTIAL_ROWWISE_LAMB,
            ]
        ),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_optimizers_lamb(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        optimizer: OptimType,
        long_segments: bool,
        pooling_mode: PoolingMode,
        use_cpu: bool,
    ) -> None:
        self.execute_backward_optimizers_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            False,  # mixed_B
            optimizer,
            long_segments,
            pooling_mode,
            use_cpu,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        optimizer=st.just(OptimType.LARS_SGD),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_optimizers_lars(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        optimizer: OptimType,
        long_segments: bool,
        pooling_mode: PoolingMode,
        use_cpu: bool,
    ) -> None:
        self.execute_backward_optimizers_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            False,  # mixed_B
            optimizer,
            long_segments,
            pooling_mode,
            use_cpu,
        )

    def execute_nbit_forward_(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        pooling_mode: PoolingMode,
        weights_ty: SparseType,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        use_cpu: bool,
        use_array_for_index_remapping: bool,
        do_pruning: bool,
        mixed_weights_ty: bool,
        output_dtype: SparseType,
    ) -> None:
        # NOTE: weighted operation can be done only for SUM.
        assume(pooling_mode == PoolingMode.SUM or not weighted)
        assume(not mixed or pooling_mode != PoolingMode.NONE)

        mode = "sum"
        do_pooling = True
        if pooling_mode == PoolingMode.SUM:
            mode = "sum"
        elif pooling_mode == PoolingMode.MEAN:
            mode = "mean"
        else:
            mode = "sum"
            do_pooling = False
        E = int(10**log_E)

        if not mixed_weights_ty:
            weights_ty_list = [weights_ty] * T
        else:
            weights_ty_list = [
                np.random.choice(
                    [
                        SparseType.FP32,
                        SparseType.FP16,
                        SparseType.FP8,
                        SparseType.INT8,
                        SparseType.INT4,
                        SparseType.INT2,
                    ]
                )
                for _ in range(T)
            ]

        D_alignment = max(
            1 if ty.bit_rate() % 8 == 0 else int(8 / ty.bit_rate())
            for ty in weights_ty_list
        )
        D = round_up(D, D_alignment)

        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(
                    np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                    D_alignment,
                )
                for _ in range(T)
            ]
            Ds = [min(D, 128) for D in Ds]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]

        if do_pooling:
            bs = [
                to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]
        else:
            bs = [
                to_device(torch.nn.Embedding(E, D, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]

        if use_cpu:
            managed = [EmbeddingLocation.HOST] * T
        elif use_cache:
            managed = [
                EmbeddingLocation.MANAGED_CACHING,
            ] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        EmbeddingLocation.DEVICE if d < average_D else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        EmbeddingLocation.DEVICE,
                        EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]

        # Fix exponent bias to 7 for now (TODO: Randomize it from a range of integers)
        if SparseType.FP8 in weights_ty_list:
            fp8_config = FP8QuantizationConfig(random.choice([4, 5]), 7)
            has_fp8_weight = True
        else:
            has_fp8_weight = False

        xs = [to_device(torch.randint(low=0, high=e, size=(B, L)), use_cpu) for e in Es]
        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(T)]

        xws_acc_type = copy.deepcopy(xws)

        if do_pruning:
            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

            (indices, offsets) = get_table_batched_offsets_from_dense(
                x, use_cpu=use_cpu
            )

            # generate index_remapping
            dense_indices = torch.randint(low=0, high=E, size=(T, B, L)).view(-1).int()

            original_E = E
            current_device = "cpu" if use_cpu else torch.cuda.current_device()

            indices = indices.view(-1).int()
            offsets = offsets.view(-1).int()

            # generate index_remapping done
            # Initialize and insert Array index remapping based data structure
            index_remappings_array = []
            for t in range(T):
                # pyre-fixme[6]: For 1st param expected `dtype` but got `Union[int,
                #  str]`.
                indice_t = (indices.view(T, B, L))[t].long().view(-1).to(current_device)
                dense_indice_t = (
                    (dense_indices.view(T, B, L))[t].view(-1)
                    # pyre-fixme[6]: For 1st param expected `dtype` but got `Union[int,
                    #  str]`.
                    .to(current_device)
                )
                index_remappings_array_t = torch.tensor(
                    [-1] * original_E,
                    dtype=torch.int32,
                    device=current_device,
                )
                index_remappings_array_t[indice_t] = dense_indice_t
                index_remappings_array.append(index_remappings_array_t.cpu())
        else:
            index_remappings_array = [torch.arange(E, dtype=torch.int32) for E in Es]
            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)
            (indices, offsets) = get_table_batched_offsets_from_dense(
                x, use_cpu=use_cpu
            )

        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    W_TY,
                    EmbeddingLocation(M),
                )
                for (E, D, M, W_TY) in zip(Es, Ds, managed, weights_ty_list)
            ],
            pooling_mode=pooling_mode,
            index_remapping=index_remappings_array if B != 0 else None,
            device="cpu" if use_cpu else torch.cuda.current_device(),
            cache_algorithm=cache_algorithm,
            use_array_for_index_remapping=use_array_for_index_remapping,
            output_dtype=output_dtype,
            fp8_exponent_bits=fp8_config.get("exponent_bits")
            if has_fp8_weight
            else None,
            fp8_exponent_bias=fp8_config.get("exponent_bias")
            if has_fp8_weight
            else None,
        )
        # Initialize the random weights for int nbit table split embedding bag
        cc.fill_random_weights()
        # NOTE: test TorchScript-compatible!
        cc = torch.jit.script(cc)

        for t in range(T):
            (weights, scale_shift) = cc.split_embedding_weights()[t]
            if scale_shift is not None:
                (E, R) = scale_shift.shape
                self.assertEqual(R, 4)
                if weights_ty_list[t] == SparseType.INT2:
                    scales = np.random.uniform(0.1, 1, size=(E,)).astype(np.float16)
                    shifts = np.random.uniform(-2, 2, size=(E,)).astype(np.float16)
                if weights_ty_list[t] == SparseType.INT4:
                    scales = np.random.uniform(0.01, 0.1, size=(E,)).astype(np.float16)
                    shifts = np.random.uniform(-2, 2, size=(E,)).astype(np.float16)
                if weights_ty_list[t] == SparseType.INT8:
                    scales = np.random.uniform(0.001, 0.01, size=(E,)).astype(
                        np.float16
                    )
                    shifts = np.random.uniform(-2, 2, size=(E,)).astype(np.float16)

                scale_shift[:, :] = torch.tensor(
                    np.stack([scales, shifts], axis=1).astype(np.float16).view(np.uint8)
                )

            fake_quantize_embs(
                weights,
                scale_shift,
                bs[t].weight.detach(),
                weights_ty_list[t],
                use_cpu=False,
                # pyre-fixme[61]: `fp8_config` is undefined, or not always defined.
                fp8_config=fp8_config if has_fp8_weight else None,
            )

        if not use_cpu:
            fc2 = (
                cc(indices.int(), offsets.int())
                if not weighted
                else cc(indices.int(), offsets.int(), xw.contiguous().view(-1))
            )
        else:
            cc = cc.cpu()
            indices, offsets = indices.cpu(), offsets.cpu()
            fc2 = (
                cc(indices.int(), offsets.int())
                if not weighted
                else cc(indices.int(), offsets.int(), xw.contiguous().view(-1).cpu())
            )

        if do_pooling and B == 0:
            self.assertEqual(fc2.size(), (0, cc.total_D))
            return

        new_indices = []
        for t in range(T):
            new_indices_t = torch.zeros([B, L], dtype=torch.int32)
            for i in range(B):
                for j in range(L):
                    old_index = xs[t][i, j]
                    new_index = index_remappings_array[t][old_index]
                    new_indices_t[i][j] = new_index
            new_indices.append(new_indices_t)

        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, new_indices)
            ]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, new_indices, xws)
            ]
        )
        if do_pooling:
            f = torch.cat([f.view(B, -1) for f in fs], dim=1)
        else:
            f = torch.cat(fs, dim=0).view(-1, D)
        torch.testing.assert_close(
            fc2.float().cpu(),
            f.float().cpu(),
            atol=1.0e-2,
            rtol=1.0e-2,
        )

    @given(
        nbit_weights_ty=get_nbit_weights_ty(),
        use_array_for_index_remapping=st.booleans(),
        do_pruning=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
    )
    def test_nbit_forward_cpu(
        self,
        nbit_weights_ty: Optional[SparseType],
        use_array_for_index_remapping: bool,
        do_pruning: bool,
    ) -> None:
        use_cpu = True
        T = random.randint(1, 50)
        B = random.randint(0, 128)
        L = random.randint(0, 32)
        D = random.randint(2, 2048)
        log_E = random.randint(2, 4)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        )
        mixed = random.choice([True, False])
        if pooling_mode == PoolingMode.NONE:
            nbit_weights_ty = random.choice(
                [
                    SparseType.FP32,
                    SparseType.FP16,
                    # CPU sequence embedding does not support FP8/INT4/INT2 yet
                    # SparseType.FP8,
                    SparseType.INT8,
                    # SparseType.INT4,
                    # SparseType.INT2,
                ]
            )

        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False

        if nbit_weights_ty is None:
            # don't care when mixed type is used.
            weights_ty: SparseType = SparseType.INT8
            mixed_weights_ty = True
        else:
            weights_ty: SparseType = nbit_weights_ty
            mixed_weights_ty = False
        output_dtype = random.choice(
            (
                [SparseType.BF16]
                if weights_ty in [SparseType.INT4, SparseType.INT2]
                else []
            )
            + [SparseType.FP32, SparseType.FP16]
        )
        self.execute_nbit_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            pooling_mode,
            weights_ty,
            use_cache,
            cache_algorithm,
            use_cpu,
            use_array_for_index_remapping,
            do_pruning,
            mixed_weights_ty,
            output_dtype,
        )

    @given(
        nbit_weights_ty=get_nbit_weights_ty(),
        use_array_for_index_remapping=st.booleans(),
        do_pruning=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
    )
    def test_nbit_forward_cpu_bf16_out(
        self,
        nbit_weights_ty: Optional[SparseType],
        use_array_for_index_remapping: bool,
        do_pruning: bool,
    ) -> None:
        use_cpu = True
        T = random.randint(1, 50)
        B = random.randint(0, 128)
        L = random.randint(0, 32)
        D = random.randint(2, 2048)
        log_E = random.randint(2, 4)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
        )
        mixed = random.choice([True, False])

        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False

        if nbit_weights_ty is None:
            # don't care when mixed type is used.
            weights_ty: SparseType = SparseType.INT8
            mixed_weights_ty = True
        else:
            weights_ty: SparseType = nbit_weights_ty
            mixed_weights_ty = False
        output_dtype = SparseType.BF16
        self.execute_nbit_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            pooling_mode,
            weights_ty,
            use_cache,
            cache_algorithm,
            use_cpu,
            use_array_for_index_remapping,
            do_pruning,
            mixed_weights_ty,
            output_dtype,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        nbit_weights_ty=get_nbit_weights_ty(),
        use_array_for_index_remapping=st.booleans(),
        do_pruning=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
    )
    def test_nbit_forward_gpu_no_cache(
        self,
        nbit_weights_ty: Optional[SparseType],
        use_array_for_index_remapping: bool,
        do_pruning: bool,
    ) -> None:
        use_cpu = False
        T = random.randint(1, 50)
        B = random.randint(0, 128)
        L = random.randint(0, 32)
        D = random.randint(2, 2048)
        log_E = random.randint(2, 4)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
        else:
            mixed = random.choice([True, False])
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False

        if nbit_weights_ty is None:
            # don't care when mixed type is used.
            weights_ty: SparseType = SparseType.INT8
            mixed_weights_ty = True
        else:
            weights_ty: SparseType = nbit_weights_ty
            mixed_weights_ty = False
        output_dtype = random.choice(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16]
        )
        self.execute_nbit_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            pooling_mode,
            weights_ty,
            use_cache,
            cache_algorithm,
            use_cpu,
            use_array_for_index_remapping,
            do_pruning,
            mixed_weights_ty,
            output_dtype,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        ),
        emulate_pruning=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_int_nbit_split_embedding_uvm_caching_codegen_lookup_function(
        self,
        weights_ty: SparseType,
        emulate_pruning: bool,
    ) -> None:
        # TODO: support direct-mapped in int_nbit_split_embedding_uvm_caching_codegen_lookup_function
        # This test is for int_nbit_split_embedding_uvm_caching_codegen_lookup_function.
        # We run IntNBitTableBatchedEmbeddingBagsCodegen with UVM_CACHING, and then
        # run int_nbit_split_embedding_uvm_caching_codegen_lookup_function with the
        # exact same cache configuration. As both use the same logic, the result
        # as well as cache state should match.

        # Currently, int_nbit_split_embedding_uvm_caching_codegen_lookup_function supports only LRU.
        cache_algorithm = CacheAlgorithm.LRU
        associativity = 32  # Currently, hard-coded 32-way set associative.
        current_device: torch.device = torch.device(torch.cuda.current_device())

        T = random.randint(1, 5)
        B = random.randint(1, 128)
        L = random.randint(1, 20)
        D = random.randint(2, 256)
        log_E = random.randint(3, 5)

        iters = 3
        E = int(10**log_E)

        D_alignment = (
            1 if weights_ty.bit_rate() % 8 == 0 else int(8 / weights_ty.bit_rate())
        )
        D = round_up(D, D_alignment)

        # Currently, int_nbit_split_embedding_uvm_caching_codegen_lookup_function supports only all UVM or all UVM_CACHING.
        Ds = [D] * T
        Es = [E] * T
        managed_caching = [EmbeddingLocation.MANAGED_CACHING] * T

        # Note both cc_ref and cc use caching.
        cc_ref = IntNBitTableBatchedEmbeddingBagsCodegen(
            [("", E, D, weights_ty, M) for (E, D, M) in zip(Es, Ds, managed_caching)],
            cache_algorithm=cache_algorithm,
        )
        cc_ref.fill_random_weights()

        # cc is only for cache states; we test int_nbit_split_embedding_uvm_caching_codegen_lookup_function directly;
        # hence, no need to synchronize cc's weights with cc_ref's.
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            [("", E, D, weights_ty, M) for (E, D, M) in zip(Es, Ds, managed_caching)],
            cache_algorithm=cache_algorithm,
        )
        cc.fill_random_weights()

        # weights_placement for all UVM case.
        managed_uvm = [EmbeddingLocation.MANAGED] * T
        placement_uvm = torch.tensor(
            managed_uvm, device=current_device, dtype=torch.int32
        )

        # zero size HBM cache for UVM case.
        zero_size_cache_weights = torch.zeros(
            0, 0, device=current_device, dtype=torch.uint8
        )

        requests = generate_requests(
            iters, B, T, L, min(Es), reuse=0.1, emulate_pruning=emulate_pruning
        )
        for indices, offsets, _ in requests:
            indices = indices.int()
            offsets = offsets.int()
            output_ref = cc_ref(indices, offsets)

            # int_nbit_split_embedding_uvm_caching_codegen_lookup_function for UVM_CACHING.
            # using weights and other params from cc_ref, but
            # cache states from cc.
            output_uvm_caching = torch.ops.fbgemm.int_nbit_split_embedding_uvm_caching_codegen_lookup_function(
                dev_weights=cc_ref.weights_host
                if cc_ref.host_size > 0
                else cc_ref.weights_dev,
                uvm_weights=cc_ref.weights_uvm,
                weights_placements=cc_ref.weights_placements,
                weights_offsets=cc_ref.weights_offsets,
                weights_tys=cc_ref.weights_tys,
                D_offsets=cc_ref.D_offsets,
                total_D=cc_ref.total_D,
                max_int2_D=cc_ref.max_int2_D,
                max_int4_D=cc_ref.max_int4_D,
                max_int8_D=cc_ref.max_int8_D,
                max_float16_D=cc_ref.max_float16_D,
                max_float32_D=cc_ref.max_float32_D,
                indices=indices,
                offsets=offsets,
                pooling_mode=int(cc_ref.pooling_mode),
                indice_weights=None,
                output_dtype=cc_ref.output_dtype,
                lxu_cache_weights=cc.lxu_cache_weights,  # cc, not cc_ref.
                lxu_cache_locations=torch.empty(0, dtype=torch.int32).fill_(-1),
                row_alignment=cc_ref.row_alignment,
                max_float8_D=cc_ref.max_float8_D,
                fp8_exponent_bits=cc_ref.fp8_exponent_bits,
                fp8_exponent_bias=cc_ref.fp8_exponent_bias,
                # Additional args for UVM_CACHING: using cc, not cc_ref.
                cache_hash_size_cumsum=cc.cache_hash_size_cumsum,
                total_cache_hash_size=cc.total_cache_hash_size,
                cache_index_table_map=cc.cache_index_table_map,
                lxu_cache_state=cc.lxu_cache_state,
                lxu_state=cc.lxu_state,
            )
            torch.testing.assert_close(output_uvm_caching, output_ref, equal_nan=True)
            # cache status; we use the exact same logic, but still assigning ways in a associative cache can be
            # arbitrary. We compare sum along ways in each set, instead of expecting exact tensor match.
            cache_weights_ref = torch.reshape(
                # pyre-fixme[6]: For 1st param expected `Tensor` but got
                #  `Union[Tensor, Module]`.
                cc_ref.lxu_cache_weights,
                [-1, associativity],
            )
            # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Tensor,
            #  Module]`.
            cache_weights = torch.reshape(cc.lxu_cache_weights, [-1, associativity])
            torch.testing.assert_close(
                torch.sum(cache_weights_ref, 1),
                torch.sum(cache_weights, 1),
                equal_nan=True,
            )
            torch.testing.assert_close(
                # pyre-fixme[6]: For 1st param expected `Tensor` but got
                #  `Union[Tensor, Module]`.
                torch.sum(cc.lxu_cache_state, 1),
                # pyre-fixme[6]: For 1st param expected `Tensor` but got
                #  `Union[Tensor, Module]`.
                torch.sum(cc_ref.lxu_cache_state, 1),
                equal_nan=True,
            )
            # lxu_state can be different as time_stamp values can be different.
            # we check the entries with max value.
            # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Tensor,
            #  Module]`.
            max_timestamp_ref = torch.max(cc_ref.lxu_state)
            # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Tensor,
            #  Module]`.
            max_timestamp_uvm_caching = torch.max(cc.lxu_state)
            x = cc_ref.lxu_state == max_timestamp_ref
            y = cc.lxu_state == max_timestamp_uvm_caching
            # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[bool,
            #  Tensor]`.
            torch.testing.assert_close(torch.sum(x, 1), torch.sum(y, 1))

            # int_nbit_split_embedding_uvm_caching_codegen_lookup_function for UVM.
            output_uvm = torch.ops.fbgemm.int_nbit_split_embedding_uvm_caching_codegen_lookup_function(
                dev_weights=cc_ref.weights_host
                if cc_ref.host_size > 0
                else cc_ref.weights_dev,
                uvm_weights=cc_ref.weights_uvm,
                weights_placements=placement_uvm,  # all UVM weights placement.
                weights_offsets=cc_ref.weights_offsets,
                weights_tys=cc_ref.weights_tys,
                D_offsets=cc_ref.D_offsets,
                total_D=cc_ref.total_D,
                max_int2_D=cc_ref.max_int2_D,
                max_int4_D=cc_ref.max_int4_D,
                max_int8_D=cc_ref.max_int8_D,
                max_float16_D=cc_ref.max_float16_D,
                max_float32_D=cc_ref.max_float32_D,
                indices=indices,
                offsets=offsets,
                pooling_mode=int(cc_ref.pooling_mode),
                indice_weights=None,
                output_dtype=cc_ref.output_dtype,
                lxu_cache_weights=zero_size_cache_weights,  # empty HBM cache.
                lxu_cache_locations=torch.empty(0, dtype=torch.int32).fill_(-1),
                row_alignment=cc_ref.row_alignment,
                max_float8_D=cc_ref.max_float8_D,
                fp8_exponent_bits=cc_ref.fp8_exponent_bits,
                fp8_exponent_bias=cc_ref.fp8_exponent_bias,
                # Additional args for UVM_CACHING; not needed for UVM.
                cache_hash_size_cumsum=None,
                total_cache_hash_size=None,
                cache_index_table_map=None,
                lxu_cache_state=None,
                lxu_state=None,
            )
            torch.testing.assert_close(output_uvm, output_ref, equal_nan=True)

    @unittest.skipIf(*gpu_unavailable)
    @given(
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        ),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        associativity=st.sampled_from([1, DEFAULT_ASSOC]),
        do_pruning=st.booleans(),
        use_array_for_index_remapping=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_forward_uvm_cache(
        self,
        weights_ty: SparseType,
        cache_algorithm: CacheAlgorithm,
        associativity: int,
        do_pruning: bool,
        use_array_for_index_remapping: bool,
    ) -> None:
        assume(cache_algorithm == CacheAlgorithm.LRU or associativity != 1)

        T = random.randint(1, 5)
        B = random.randint(1, 128)
        L = random.randint(1, 20)
        D = random.randint(2, 256)
        log_E = random.randint(3, 5)
        mixed = random.choice([True, False])

        iters = 3
        E = int(10**log_E)

        D_alignment = (
            1 if weights_ty.bit_rate() % 8 == 0 else int(8 / weights_ty.bit_rate())
        )
        D = round_up(D, D_alignment)

        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(
                    np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                    D_alignment,
                )
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        managed = [EmbeddingLocation.MANAGED_CACHING] * T
        if mixed:
            average_D = sum(Ds) // T
            for t, d in enumerate(Ds):
                managed[t] = EmbeddingLocation.DEVICE if d < average_D else managed[t]
        index_remapping = None
        pruning_hash_load_factor = 0.5
        if do_pruning:
            current_device = torch.cuda.current_device()
            index_remapping = []
            for E in Es:
                # For each table, keep the first half of rows as is, but
                # the rest is treated as pruned (-1).
                remapping = list(range(0, E // 2)) + [-1] * (E - E // 2)
                remapping_t = torch.tensor(
                    remapping,
                    dtype=torch.int32,
                    device=current_device,
                )
                index_remapping.append(remapping_t)
        cc_ref = IntNBitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    "",
                    E,
                    D,
                    weights_ty,
                    EmbeddingLocation.DEVICE,
                )
                for (E, D) in zip(Es, Ds)
            ],
            index_remapping=index_remapping,
            use_array_for_index_remapping=use_array_for_index_remapping,
            pruning_hash_load_factor=pruning_hash_load_factor,
        )
        cc_ref.fill_random_weights()
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            [("", E, D, weights_ty, M) for (E, D, M) in zip(Es, Ds, managed)],
            cache_algorithm=cache_algorithm,
            cache_assoc=associativity,
            index_remapping=index_remapping,
            use_array_for_index_remapping=use_array_for_index_remapping,
            pruning_hash_load_factor=pruning_hash_load_factor,
        )
        cc.fill_random_weights()

        split_weights = cc.split_embedding_weights()
        ref_split_weights = cc_ref.split_embedding_weights()
        for t in range(T):
            (weights, scale_shift) = split_weights[t]
            (ref_weights, ref_scale_shift) = ref_split_weights[t]
            self.assertEqual(weights.size(), ref_weights.size())
            weights.copy_(ref_weights)
            if ref_scale_shift is not None:
                scale_shift.copy_(ref_scale_shift)

        requests = generate_requests(iters, B, T, L, min(Es), reuse=0.1)

        for indices, offsets, _ in requests:
            indices = indices.int()
            offsets = offsets.int()
            output = cc(indices, offsets)
            output_ref = cc_ref(indices, offsets)
            torch.testing.assert_close(output, output_ref, equal_nan=True)

    @given(
        T=st.integers(min_value=1, max_value=5),
        B=st.integers(min_value=1, max_value=8),
        L=st.integers(min_value=0, max_value=8),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        use_cpu_hashtable=st.booleans(),
        use_array_for_index_remapping=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_pruning(
        self,
        T: int,
        B: int,
        L: int,
        use_cpu: bool,
        use_cpu_hashtable: bool,
        use_array_for_index_remapping: bool,
    ) -> None:
        E = int(1000)
        LOAD_FACTOR = 0.8
        pruning_ratio = 0.5

        capacities = [int(B * L / LOAD_FACTOR) + 1 for _ in range(T)]
        original_E = int(E / (1.0 - pruning_ratio))

        # Enforce the size of original_E/B/L to get the unique indices
        assume(original_E > B * L)

        current_device = "cpu" if use_cpu else torch.cuda.current_device()

        if use_cpu_hashtable:
            assume(use_cpu)

        indices = torch.randint(low=0, high=original_E, size=(T, B, L))
        for t in range(T):
            while (
                torch.unique(
                    indices[t], return_counts=False, return_inverse=False
                ).numel()
                != indices[t].numel()
            ):
                indices[t] = torch.randint(low=0, high=original_E, size=(B, L))

        indices = indices.view(-1).int()
        dense_indices = torch.randint(low=0, high=E, size=(T, B, L)).view(-1).int()
        offsets = torch.tensor([L * b_t for b_t in range(B * T + 1)]).int()

        # Initialize and insert Hashmap index remapping based data structure
        hash_table = torch.empty(
            (sum(capacities), 2),
            dtype=torch.int32,
        )
        hash_table[:, :] = -1
        hash_table_offsets = torch.tensor([0] + np.cumsum(capacities).tolist()).long()

        torch.ops.fbgemm.pruned_hashmap_insert(
            indices, dense_indices, offsets, hash_table, hash_table_offsets
        )

        if use_cpu_hashtable:
            ht = torch.classes.fbgemm.PrunedMapCPU()
            ht.insert(indices, dense_indices, offsets, T)

        # Initialize and insert Array index remapping based data structure
        index_remappings_array = torch.tensor(
            [-1] * original_E * T,
            dtype=torch.int32,
            device=current_device,
        )
        index_remappings_array_offsets = torch.empty(
            T + 1,
            dtype=torch.int64,
            # pyre-fixme[6]: For 3rd param expected `Union[None, str, device]` but
            #  got `Union[int, str]`.
            device=current_device,
        )
        index_remappings_array_offsets[0] = 0
        for t in range(T):
            # pyre-fixme[6]: For 1st param expected `dtype` but got `Union[int, str]`.
            indice_t = (indices.view(T, B, L))[t].long().view(-1).to(current_device)
            dense_indice_t = (
                (dense_indices.view(T, B, L))[t].view(-1)
                # pyre-fixme[6]: For 1st param expected `dtype` but got `Union[int,
                #  str]`.
                .to(current_device)
            )
            selected_indices = torch.add(indice_t, t * original_E)[:E]
            index_remappings_array[selected_indices] = dense_indice_t
            index_remappings_array_offsets[t + 1] = (
                index_remappings_array_offsets[t] + original_E
            )

        # Move data when using device
        if not use_cpu:
            (
                indices,
                dense_indices,
                offsets,
                hash_table,
                hash_table_offsets,
                index_remappings_array,
                index_remappings_array_offsets,
            ) = (
                # pyre-fixme[6]: For 1st param expected `dtype` but got `Union[int,
                #  str]`.
                indices.to(current_device),
                # pyre-fixme[6]: For 1st param expected `dtype` but got `Union[int,
                #  str]`.
                dense_indices.to(current_device),
                # pyre-fixme[6]: For 1st param expected `dtype` but got `Union[int,
                #  str]`.
                offsets.to(current_device),
                # pyre-fixme[6]: For 1st param expected `dtype` but got `Union[int,
                #  str]`.
                hash_table.to(current_device),
                # pyre-fixme[6]: For 1st param expected `dtype` but got `Union[int,
                #  str]`.
                hash_table_offsets.to(current_device),
                # pyre-fixme[6]: For 1st param expected `dtype` but got `Union[int,
                #  str]`.
                index_remappings_array.to(current_device),
                # pyre-fixme[6]: For 1st param expected `dtype` but got `Union[int,
                #  str]`.
                index_remappings_array_offsets.to(current_device),
            )

        # Lookup
        if use_cpu_hashtable:
            dense_indices_ = ht.lookup(indices, offsets)
        elif not use_array_for_index_remapping:  # hashmap based pruning
            dense_indices_ = torch.ops.fbgemm.pruned_hashmap_lookup(
                indices, offsets, hash_table, hash_table_offsets
            )
        else:  # array based pruning
            dense_indices_ = torch.ops.fbgemm.pruned_array_lookup(
                indices,
                offsets,
                index_remappings_array,
                index_remappings_array_offsets,
            )

        # Validate the lookup result
        torch.testing.assert_close(dense_indices, dense_indices_)

        # For array based pruning, it will be out-of-boundary for arbitrarily
        # large indices. We will rely on bound checker to make sure indices
        # are within the boundary.
        if not use_array_for_index_remapping:
            # now, use a value that does not exist in the original set of indices
            # and so should be pruned out.
            indices[:] = np.iinfo(np.int32).max

            if use_cpu_hashtable:
                dense_indices_ = ht.lookup(indices, offsets)
            elif not use_array_for_index_remapping:  # hashmap based pruning
                dense_indices_ = torch.ops.fbgemm.pruned_hashmap_lookup(
                    indices, offsets, hash_table, hash_table_offsets
                )
            else:  # array based pruning
                dense_indices_ = torch.ops.fbgemm.pruned_array_lookup(
                    indices,
                    offsets,
                    index_remappings_array,
                    index_remappings_array_offsets,
                )
            torch.testing.assert_close(dense_indices.clone().fill_(-1), dense_indices_)

    @given(
        L=st.integers(min_value=0, max_value=16),
        H=st.integers(min_value=512, max_value=1024),
        S=st.integers(min_value=0, max_value=128),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
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

    @given(N=st.integers(min_value=1, max_value=8))
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
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

    @given(N=st.integers(min_value=1, max_value=2))
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_stb_uvm_cache_stats(self, N: int) -> None:
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
            gather_uvm_cache_stats=True,
        )

        x = torch.Tensor([[[1], [1]], [[3], [4]]])
        x = to_device(torch.tensor(x, dtype=torch.int64), use_cpu=False)

        for _ in range(N):
            indices, offsets = get_table_batched_offsets_from_dense(x, use_cpu=False)
            cc.reset_cache_states()
            cc.reset_uvm_cache_stats()
            cc(indices, offsets)
            (
                n_calls,
                n_requested_indices,
                n_unique_indices,
                n_unique_misses,
                n_conflict_unique_misses,
                n_conflict_misses,
            ) = cc.get_uvm_cache_stats()
            self.assertEqual(n_calls, 1)
            self.assertEqual(n_requested_indices, len(indices))
            self.assertEqual(n_unique_indices, len(set(indices.tolist())))
            self.assertEqual(n_unique_misses, len(set(indices.tolist())))
            self.assertEqual(n_conflict_unique_misses, 0)
            self.assertEqual(n_conflict_misses, 0)

    @unittest.skipIf(*gpu_unavailable)
    @given(
        L=st.integers(min_value=0, max_value=16),
        H=st.integers(min_value=512, max_value=1024),
        S=st.integers(min_value=0, max_value=128),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_cache_update_function(self, L: int, H: int, S: int) -> None:
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
        linear_cache_indices = linear_cache_indices_cpu.to(torch.int32).cuda()
        lxu_cache_locations = lxu_cache_locations_cpu.to(torch.int32).cuda()
        expected_unique_access = len(torch.unique(linear_cache_indices_cpu))
        expected_total_access = len(linear_cache_indices_cpu)

        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            record_cache_metrics=RecordCacheMetrics(True, False),
        )
        cc.fill_random_weights()

        cc._update_cache_miss_counter(lxu_cache_locations, linear_cache_indices)
        (
            cache_miss_forward_count,
            unique_cache_miss_count,
            unique_access_count,
            total_access_count,
        ) = cc.get_cache_miss_counter().cpu()

        self.assertEqual(unique_cache_miss_count, expect_out)
        self.assertLessEqual(cache_miss_forward_count, unique_cache_miss_count)
        self.assertEqual(unique_access_count, expected_unique_access)
        self.assertEqual(total_access_count, expected_total_access)

    @unittest.skipIf(*gpu_unavailable)
    @given(N=st.integers(min_value=1, max_value=8))
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_cache_miss_counter(self, N: int) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            record_cache_metrics=RecordCacheMetrics(True, True),
        )
        cc.fill_random_weights()

        # Create fake input data and the target output
        x1 = torch.Tensor([[[1], [1]], [[3], [4]]]).cuda()
        x2 = torch.Tensor([[[2], [1]], [[3], [4]]]).cuda()
        x3 = torch.Tensor([[[5], [6]], [[7], [8]]]).cuda()

        xs = [x1, x2, x3]
        target_counter_list = [[1, 3], [2, 4], [3, 8]]
        target_tablewise_cache_miss_list = [[1, 2], [2, 2], [4, 4]]
        for x, t_counter, t_tablewise_cache_miss in zip(
            xs, target_counter_list, target_tablewise_cache_miss_list
        ):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                cc(indices.int(), offsets.int())
                (
                    cache_miss_forward_count,
                    unique_cache_miss_count,
                    _,
                    _,
                ) = cc.get_cache_miss_counter().cpu()
                tablewise_cache_miss = cc.get_table_wise_cache_miss().cpu()
                self.assertEqual(cache_miss_forward_count, t_counter[0])
                self.assertEqual(unique_cache_miss_count, t_counter[1])
                for i in range(len(tablewise_cache_miss)):
                    self.assertEqual(tablewise_cache_miss[i], t_tablewise_cache_miss[i])

    @unittest.skipIf(*gpu_unavailable)
    @given(
        N=st.integers(min_value=1, max_value=8),
        dtype=st.sampled_from([SparseType.INT8, SparseType.INT4, SparseType.INT2]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_uvm_cache_stats(self, N: int, dtype: SparseType) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    dtype,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            gather_uvm_cache_stats=True,
        )
        cc.fill_random_weights()

        # Create fake input data and the target output
        x1 = torch.Tensor([[[1], [1]], [[3], [4]]]).cuda()
        x2 = torch.Tensor([[[2], [1]], [[3], [4]]]).cuda()
        x3 = torch.Tensor([[[5], [6]], [[7], [8]]]).cuda()

        xs = [x1, x2, x3]
        # num_unique_indices, num_unique_misses
        # note that these are cumulative over calls; and also "unique" is per batch.
        target_counter_list = [[3, 3], [4, 4], [4, 8]]
        num_calls_expected = 0
        num_indices_expcted = 0
        num_unique_indices_expected = 0
        for x, t_counter in zip(xs, target_counter_list):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                num_calls_expected = num_calls_expected + 1
                num_indices_expcted = num_indices_expcted + len(indices)
                cc(indices.int(), offsets.int())
                (
                    num_calls,
                    num_indices,
                    num_unique_indices,
                    num_unique_misses,
                    num_conflict_unique_miss,
                    num_conflict_miss,
                ) = cc.get_uvm_cache_stats().cpu()
                # Note num_unique_indices is cumulative stats.
                num_unique_indices_expected = num_unique_indices_expected + t_counter[0]
                self.assertEqual(num_calls, num_calls_expected)
                self.assertEqual(num_indices, num_indices_expcted)
                self.assertEqual(num_unique_indices, num_unique_indices_expected)
                self.assertEqual(num_unique_misses, t_counter[1])
                self.assertEqual(num_conflict_unique_miss, 0)
                self.assertEqual(num_conflict_miss, 0)

        T = 1  # for simplicity
        Ds = [D] * T
        Es = [E] * T
        cc1 = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            gather_uvm_cache_stats=True,
            cache_sets=1,  # Only one set.
        )
        cc1.fill_random_weights()

        associativty = DEFAULT_ASSOC  # 32 for NVidia / 64 for AMD.
        repetition = 17
        indices1 = torch.Tensor(
            [[list(range(0, associativty))] * repetition]
        ).cuda()  # 0, 1, ..., 31.
        indices2 = torch.Tensor(
            [[list(range(0, associativty + 1))] * repetition]
        ).cuda()  # 0, 1, ..., 31, 32.
        indices3 = torch.Tensor(
            [[list(range(0, associativty + 10))] * repetition]
        ).cuda()  # 0, 1, ..., 31, 32, ..., 41.

        # num_conflict_unique_miss, num_conflict_miss
        expected = [[0, 0], [1, 17], [10, 170]]

        for x, e in zip((indices1, indices2, indices3), expected):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                cc1(indices.int(), offsets.int())
                (
                    _,
                    _,
                    _,
                    _,
                    num_conflict_unique_miss,
                    num_conflict_miss,
                ) = cc1.get_uvm_cache_stats().cpu()
                self.assertEqual(num_conflict_unique_miss, e[0])
                self.assertEqual(num_conflict_miss, e[1])
                cc1.reset_uvm_cache_stats()

    @given(
        T=st.integers(min_value=1, max_value=64),
        B=st.integers(min_value=1, max_value=64),
        max_L=st.integers(min_value=1, max_value=64),
        bounds_check_mode=st.sampled_from(
            [
                BoundsCheckMode.FATAL,
                BoundsCheckMode.WARNING,
                BoundsCheckMode.IGNORE,
            ]
        ),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        weighted=st.booleans(),
        dtype=st.sampled_from(
            [
                torch.int64,
                torch.int32,
            ]
        ),
        mixed_B=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_bounds_check(  # noqa C901
        self,
        T: int,
        B: int,
        max_L: int,
        bounds_check_mode: BoundsCheckMode,
        use_cpu: bool,
        weighted: bool,
        dtype: torch.dtype,
        mixed_B: bool,
    ) -> None:
        # use_cpu does not support mixed_B
        if use_cpu and mixed_B:
            mixed_B = False
        rows_per_table = torch.tensor(
            np.random.randint(low=1, high=1000, size=(T,))
        ).long()
        if not mixed_B:
            Bs = [B] * T
        else:
            low = max(int(0.25 * B), 1)
            high = int(B)
            if low == high:
                Bs = [B] * T
            else:
                Bs = [np.random.randint(low=low, high=high) for _ in range(T)]
        B_offsets = [0] + list(accumulate(Bs))
        Ls = np.random.randint(low=0, high=max_L, size=(B_offsets[-1],))
        indices = [
            np.random.randint(
                low=0,
                high=rows_per_table[t],
                size=sum(Ls[B_offsets[t] : B_offsets[t + 1]]),
            )
            for t in range(T)
        ]
        indices = torch.tensor(np.concatenate(indices, axis=0)).to(dtype)
        weights = (
            torch.rand(indices.shape, dtype=torch.float, device=indices.device)
            if weighted
            else None
        )
        offsets = torch.tensor([0] + np.cumsum(Ls.flatten()).tolist()).to(dtype)
        warning = torch.tensor([0]).long()

        if mixed_B:
            B_offsets = torch.tensor(B_offsets, device="cuda", dtype=torch.int32)
            max_B = max(Bs)
        else:
            B_offsets = None
            max_B = -1

        self.assertEqual(indices.numel(), np.sum(Ls).item())
        self.assertEqual(offsets[-1], np.sum(Ls).item())
        if not use_cpu:
            indices, offsets, rows_per_table, warning = (
                indices.cuda(),
                offsets.cuda(),
                rows_per_table.cuda(),
                warning.cuda(),
            )
            if weighted:
                # pyre-fixme[16]: `Optional` has no attribute `cuda`.
                weights = weights.cuda()
        indices_copy = indices.clone()
        offsets_copy = offsets.clone()
        torch.ops.fbgemm.bounds_check_indices(
            rows_per_table,
            indices,
            offsets,
            bounds_check_mode,
            warning,
            weights,
            B_offsets=B_offsets,
            max_B=max_B,
        )
        # we don't modify when we are in-bounds.
        torch.testing.assert_close(indices_copy, indices)
        indices[:] = torch.iinfo(dtype).max
        if bounds_check_mode != BoundsCheckMode.FATAL:
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                bounds_check_mode,
                warning,
                weights,
                B_offsets=B_offsets,
                max_B=max_B,
            )
            torch.testing.assert_close(indices, torch.zeros_like(indices))
            if bounds_check_mode == BoundsCheckMode.WARNING:
                self.assertEqual(warning.item(), indices.numel())
        else:
            if use_cpu and indices.numel():
                with self.assertRaises(RuntimeError):
                    torch.ops.fbgemm.bounds_check_indices(
                        rows_per_table,
                        indices,
                        offsets,
                        bounds_check_mode,
                        warning,
                        weights,
                        B_offsets=B_offsets,
                        max_B=max_B,
                    )
            # It would be nice to test the CUDA implementation of BoundsCheckMode==FATAL,
            # but the device assert kills the CUDA context and requires a process restart,
            # which is a bit inconvenient.

        # test offsets bound errors
        indices = indices_copy.clone()
        offsets = offsets_copy.clone()
        if offsets.numel() > 0:
            offsets[0] = -100
        if offsets.numel() > 1:
            offsets[-1] += 100
        if bounds_check_mode != BoundsCheckMode.FATAL:
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                bounds_check_mode,
                warning,
                weights,
                B_offsets=B_offsets,
                max_B=max_B,
            )
            if offsets.numel() > 0:
                self.assertEqual(offsets[0].item(), 0)
            if offsets.numel() > 1:
                self.assertEqual(offsets[-1].item(), indices.numel())
            if bounds_check_mode == BoundsCheckMode.WARNING:
                # -1 because when we have 2 elements in offsets, we have only 1
                # warning for the pair.
                self.assertGreaterEqual(warning.item(), min(2, offsets.numel() - 1))
        else:
            if use_cpu and indices.numel():
                with self.assertRaises(RuntimeError):
                    torch.ops.fbgemm.bounds_check_indices(
                        rows_per_table,
                        indices,
                        offsets,
                        bounds_check_mode,
                        warning,
                        weights,
                    )

        # test offsets.size(0) ! = B * T + 1 case. Here we test with T >= 2 case.
        # If T == 1, we will always get the even division.
        # (does not apply to mixed_B = True)
        if not mixed_B and T >= 2:
            indices = indices_copy.clone()
            offsets = offsets_copy.clone()
            offsets = torch.cat(
                (
                    offsets,
                    torch.tensor(
                        [indices.numel()] * (T - 1),
                        dtype=offsets.dtype,
                        device=offsets.device,
                    ),
                ),
                dim=0,
            )
            with self.assertRaises(RuntimeError):
                torch.ops.fbgemm.bounds_check_indices(
                    rows_per_table,
                    indices,
                    offsets,
                    bounds_check_mode,
                    warning,
                    weights,
                )

        # test weights.size(0) != indices.size(0) case
        weights = torch.rand(
            (indices.size(0) + 1,), dtype=torch.float, device=indices.device
        )
        with self.assertRaises(RuntimeError):
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                bounds_check_mode,
                warning,
                weights,
                B_offsets=B_offsets,
                max_B=max_B,
            )

    def test_pickle(self) -> None:
        tensor_queue = torch.classes.fbgemm.TensorQueue(torch.empty(0))
        pickled = pickle.dumps(tensor_queue)
        unpickled = pickle.loads(pickled)  # noqa: F841

    @unittest.skipIf(*gpu_unavailable)
    def test_linearize_cache_indices(self) -> None:
        indices = torch.tensor(
            [10, 2, 3, 7, 1, 4, 5, 9, 2, 7, 6, 8, 5, 1, 0, 4],
            dtype=torch.int,
            device="cuda",
        )
        pruned_indices = torch.tensor(
            [10, -1, 3, 7, 1, 4, -1, 9, 2, -1, 6, 8, 5, 1, -1, 4],
            dtype=torch.int,
            device="cuda",
        )
        equal_offsets = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int, device="cuda")
        varying_offsets = torch.tensor(
            [0, 1, 3, 6, 8, 10, 14, 15, 16], dtype=torch.int, device="cuda"
        )

        # Testing equal sized tables.
        cache_hash_size_cumsum_0 = torch.tensor([0, 12, 24, 36, 48]).cuda()
        linear_cache_indices_0 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_0, indices, equal_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_0.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 13, 16, 17, 21, 26, 31, 30, 32, 41, 37, 36, 40],
                    dtype=torch.int,
                ),
            )
        )

        # Testing partially cached tables.
        cache_hash_size_cumsum_1 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_1 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_1, indices, equal_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_1.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 13, 16, 17, 21, 36, 36, 36, 36, 29, 25, 24, 28],
                    dtype=torch.int,
                ),
            )
        )

        # Testing batched with varying pooling factor.
        cache_hash_size_cumsum_2 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_2 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_2, indices, varying_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_2.cpu(),
                torch.tensor(
                    [10, 2, 3, 19, 13, 16, 17, 21, 36, 36, 36, 36, 36, 36, 24, 28],
                    dtype=torch.int,
                ),
            )
        )

        # Testing when multiple features share the same table.
        cache_hash_size_cumsum_3 = torch.tensor([0, 0, 12, 12, 24]).cuda()
        linear_cache_indices_3 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_3, indices, varying_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_3.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 1, 4, 5, 9, 14, 19, 18, 20, 17, 13, 12, 16],
                    dtype=torch.int,
                ),
            )
        )

        # Testing equal sized tables + pruned indices
        cache_hash_size_cumsum_4 = torch.tensor([0, 12, 24, 36, 48]).cuda()
        linear_cache_indices_4 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_4, pruned_indices, equal_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_4.cpu(),
                torch.tensor(
                    [10, 48, 3, 7, 13, 16, 48, 21, 26, 48, 30, 32, 41, 37, 48, 40],
                    dtype=torch.int,
                ),
            )
        )

        # Testing batched with varying pooling factor + pruned indices
        cache_hash_size_cumsum_5 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_5 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_5, pruned_indices, varying_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_5.cpu(),
                torch.tensor(
                    [10, 36, 3, 19, 13, 16, 36, 21, 36, 36, 36, 36, 36, 36, 36, 28],
                    dtype=torch.int,
                ),
            )
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_linearize_cache_indices_from_row_idx(self) -> None:
        update_row_indices = torch.tensor(
            [10, 2, 3, 7, 1, 4, 5, 9, 2, 7, 6, 8, 5, 1, 0, 4],
            dtype=torch.int,
            device="cuda",
        )
        update_table_indices = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            dtype=torch.int,
            device="cuda",
        )
        varying_update_table_indices = torch.tensor(
            [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3],
            dtype=torch.int,
            device="cuda",
        )

        # Testing equal sized tables.
        cache_hash_size_cumsum_0 = torch.tensor([0, 12, 24, 36, 48]).cuda()
        linear_cache_indices_0 = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
            cache_hash_size_cumsum_0,
            update_table_indices,
            update_row_indices,
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_0.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 13, 16, 17, 21, 26, 31, 30, 32, 41, 37, 36, 40],
                    dtype=torch.int,
                ),
            )
        )

        # Testing partially cached tables.
        cache_hash_size_cumsum_1 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_1 = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
            cache_hash_size_cumsum_1,
            update_table_indices,
            update_row_indices,
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_1.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 13, 16, 17, 21, 36, 36, 36, 36, 29, 25, 24, 28],
                    dtype=torch.int,
                ),
            )
        )

        # Testing batched with varying pooling factor.
        cache_hash_size_cumsum_2 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_2 = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
            cache_hash_size_cumsum_2,
            varying_update_table_indices,
            update_row_indices,
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_2.cpu(),
                torch.tensor(
                    [10, 2, 3, 19, 13, 16, 17, 21, 36, 36, 36, 36, 36, 36, 24, 28],
                    dtype=torch.int,
                ),
            )
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        associativity=st.sampled_from([1, DEFAULT_ASSOC]),
    )
    @settings(deadline=None)
    def test_lxu_cache_lookup(self, associativity: int) -> None:
        max_index: int = 8000
        # Use single cache set to avoid dealing with cache set hash algorithm.
        lxu_cache_state_gpu = (
            torch.arange(associativity, dtype=torch.int64).unsqueeze(0).cuda()
        )

        # Testing all miss.
        linear_cache_indices_0 = (
            torch.tensor([32, 33, 34, 35, 36, 100, 1000, 1725])
            if associativity <= 32
            else torch.tensor([64, 65, 66, 67, 68, 100, 1000, 1725])
        ).cuda()
        lxu_locations = torch.ops.fbgemm.lxu_cache_lookup(
            linear_cache_indices_0, lxu_cache_state_gpu, max_index
        )
        torch.testing.assert_close(
            lxu_locations,
            torch.full_like(lxu_locations, -1),
        )

        # Testing all hits.
        cache_indices_1 = torch.randint(0, associativity, (associativity,))
        linear_cache_indices_1 = cache_indices_1.cuda()
        lxu_locations = torch.ops.fbgemm.lxu_cache_lookup(
            linear_cache_indices_1, lxu_cache_state_gpu, max_index
        )
        torch.testing.assert_close(
            lxu_locations.cpu(),
            cache_indices_1.int(),
        )

        # Testing mixture.
        miss_cache_indices_0 = torch.randint(associativity, max_index // 2, (10,))
        hit_cache_indices_0 = torch.randint(0, associativity, (8,))
        miss_cache_indices_1 = torch.randint(max_index // 2, max_index, (16,))
        hit_cache_indices_1 = torch.randint(0, associativity, (8,))
        linear_cache_indices_2 = torch.cat(
            [
                miss_cache_indices_0,
                hit_cache_indices_0,
                miss_cache_indices_1,
                hit_cache_indices_1,
            ]
        ).cuda()
        lxu_locations = torch.ops.fbgemm.lxu_cache_lookup(
            linear_cache_indices_2, lxu_cache_state_gpu, max_index
        )

        expected_result = torch.cat(
            [
                torch.full_like(miss_cache_indices_0, -1),
                hit_cache_indices_0,
                torch.full_like(miss_cache_indices_1, -1),
                hit_cache_indices_1,
            ]
        ).int()
        torch.testing.assert_close(
            lxu_locations.cpu(),
            expected_result,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=64),
        log_E=st.integers(min_value=2, max_value=3),
        N=st.integers(min_value=0, max_value=50),
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        ),
        output_dtype=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
            ]
        ),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        test_internal=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_embedding_inplace_update(
        self,
        T: int,  # num of embedding tables
        D: int,  # embedding dim
        log_E: int,  # embedding table row number
        N: int,  # num of update rows per table
        weights_ty: SparseType,
        output_dtype: SparseType,
        use_cpu: bool,
        test_internal: bool,  # test with OSS op or internal customized op
    ) -> None:
        D_alignment = max(weights_ty.align_size(), output_dtype.align_size())
        D = round_up(D, D_alignment)
        Ds = [
            round_up(
                np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                D_alignment,
            )
            for _ in range(T)
        ]
        E = int(10**log_E)
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]
        row_alignment = 1 if use_cpu else 16
        current_device = "cpu" if use_cpu else torch.cuda.current_device()
        location = EmbeddingLocation.HOST if use_cpu else EmbeddingLocation.DEVICE

        weights_ty_list = [weights_ty] * T
        if open_source:
            test_internal = False

        # create two embedding bag op with random weights
        locations = [location] * T
        op = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                ("", E, D, W_TY, L)
                for (E, D, W_TY, L) in zip(Es, Ds, weights_ty_list, locations)
            ],
            output_dtype=output_dtype,
            device=current_device,
        )
        op.fill_random_weights()
        op_ref = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                ("", E, D, W_TY, L)
                for (E, D, W_TY, L) in zip(Es, Ds, weights_ty_list, locations)
            ],
            output_dtype=output_dtype,
            device=current_device,
        )
        op_ref.fill_random_weights()

        # randomly generate update table and row indices
        update_table_indices = []
        update_table_indices2 = []
        update_row_indices = []
        update_row_indices2 = []
        for t in range(T):
            n = np.random.randint(low=0, high=N) if N > 0 else 0
            if n == 0:
                continue
            update_table_indices.append(t)
            update_row_id_list = random.sample(range(Es[t]), n)
            update_row_indices.append(update_row_id_list)
            update_table_indices2.extend([t] * n)
            update_row_indices2.extend(update_row_id_list)

        # generate update tensor based on weights from "op_ref" embedding table
        update_weights_list = []
        ref_split_weights = op_ref.split_embedding_weights(split_scale_shifts=False)

        update_weight_size = sum(
            [
                rounded_row_size_in_bytes(
                    Ds[t],
                    weights_ty_list[t],
                    row_alignment,
                )
                for t in update_table_indices2
            ]
        )
        update_weights_tensor2 = torch.randint(
            low=0,
            high=255,
            size=(update_weight_size,),
            dtype=torch.uint8,
            device=current_device,
        )

        update_offsets = 0
        for i in range(len(update_table_indices)):
            table_idx = update_table_indices[i]
            (ref_weights, _) = ref_split_weights[table_idx]

            D_bytes = rounded_row_size_in_bytes(
                Ds[table_idx], weights_ty_list[table_idx], row_alignment
            )

            update_weights = []
            for row_idx in update_row_indices[i]:
                update_weights.append(ref_weights[row_idx].tolist())
                update_weights_tensor2[
                    update_offsets : update_offsets + D_bytes
                ] = ref_weights[row_idx]
                update_offsets += D_bytes

            update_weights_tensor = torch.tensor(
                update_weights,
                device=current_device,
                dtype=torch.uint8,
            )
            update_weights_list.append(update_weights_tensor)

        # run inplace update on "op" embedding table
        if not test_internal:
            # Test scatter_ based OSS solution
            op.embedding_inplace_update(
                update_table_indices,
                update_row_indices,
                update_weights_list,
            )
        else:
            # Test customized op
            op.embedding_inplace_update_internal(
                update_table_indices2,
                update_row_indices2,
                update_weights_tensor2,
            )

        # verify weights are equal with "op_ref" for the updated rows in "op"
        split_weights = op.split_embedding_weights(split_scale_shifts=False)
        for i in range(len(update_table_indices)):
            t = update_table_indices[i]
            for r in update_row_indices[i]:
                (weights, _) = split_weights[t]
                (ref_weights, _) = ref_split_weights[t]
                self.assertEqual(weights.size(), ref_weights.size())
                torch.testing.assert_close(
                    weights[r],
                    ref_weights[r],
                    rtol=1e-2,
                    atol=1e-2,
                    equal_nan=True,
                )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        log_E=st.integers(min_value=2, max_value=3),
        weights_precision=st.sampled_from(
            [SparseType.FP16, SparseType.FP32, SparseType.INT8]
        ),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        num_indices_per_table=st.integers(min_value=1, max_value=5),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES,
        deadline=None,
    )
    def test_reset_embedding_weight_momentum(
        self,
        T: int,
        D: int,
        log_E: int,
        weights_precision: SparseType,
        mixed: bool,
        use_cache: bool,
        output_dtype: SparseType,
        num_indices_per_table: int,
    ) -> None:
        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        E = int(10**log_E)
        D = D * 4
        Ds: List[int] = []
        Es: List[int] = []
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        compute_device = ComputeDevice.CUDA
        if use_cache:
            managed = [EmbeddingLocation.MANAGED_CACHING] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        EmbeddingLocation.DEVICE if d < average_D else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        EmbeddingLocation.DEVICE,
                        EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        optimizer = OptimType.EXACT_ROWWISE_ADAGRAD
        cc = emb_op(
            embedding_specs=[
                (E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)
            ],
            optimizer=optimizer,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
        )

        pruned_indices: List[int] = []
        pruned_indices_offsets: List[int] = [0]
        logical_table_ids: List[int] = []
        buffer_ids: List[int] = []
        for i in range(len(Es)):
            indices = [
                np.random.randint(low=1, high=int(Es[i] - 2))
                for _ in range(num_indices_per_table)
            ]
            pruned_indices += indices
            pruned_indices_offsets.append(
                pruned_indices_offsets[i] + num_indices_per_table
            )
            logical_table_ids.append(i)
            buffer_ids.append(i)
        pruned_indices_tensor = to_device(
            torch.tensor(pruned_indices, dtype=torch.int64, requires_grad=False), False
        )
        pruned_indices_offsets_tensor = to_device(
            torch.tensor(
                pruned_indices_offsets, dtype=torch.int64, requires_grad=False
            ),
            False,
        )
        logical_table_ids_tensor = to_device(
            torch.tensor(logical_table_ids, dtype=torch.int32, requires_grad=False),
            False,
        )
        buffer_ids_tensor = to_device(
            torch.tensor(buffer_ids, dtype=torch.int32, requires_grad=False), False
        )

        momentum1: List[Tensor] = [
            s for (s,) in cc.split_optimizer_states()
        ]  # List[rows]
        weight: List[Tensor] = cc.split_embedding_weights()  # List[(rows, dim)]
        for t in range(T):
            momentum1[t].fill_(1)
            weight[t].fill_(1)

        def check_weight_momentum(v: int) -> None:
            for i in range(len(pruned_indices)):
                logical_id = i // num_indices_per_table
                table_momentum1 = momentum1[logical_id]
                table_weight = weight[logical_id]
                dim = Ds[logical_id]
                expected_row_momentum1 = to_device(
                    torch.tensor(v, dtype=torch.float32), False
                )
                expected_row_weight = to_device(
                    torch.tensor([v] * dim, dtype=weights_precision.as_dtype()),
                    False,
                )
                pruned_index = pruned_indices[i]
                row_weight = table_weight[pruned_index]
                if weights_precision == SparseType.INT8:
                    row_weight = row_weight[:-INT8_EMB_ROW_DIM_OFFSET]
                self.assertEqual(table_momentum1[pruned_index], expected_row_momentum1)
                torch.testing.assert_close(
                    row_weight,
                    expected_row_weight,
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )

        check_weight_momentum(1)

        cc.reset_embedding_weight_momentum(
            pruned_indices_tensor,
            pruned_indices_offsets_tensor,
            logical_table_ids_tensor,
            buffer_ids_tensor,
        )

        check_weight_momentum(0)


if __name__ == "__main__":
    unittest.main()
