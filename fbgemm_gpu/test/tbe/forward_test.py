#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import copy
import random
import unittest
from typing import Callable, Dict, List

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_embedding_utils import (
    b_indices,
    generate_requests,
    get_table_batched_offsets_from_dense,
    round_up,
    to_device,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from hypothesis import assume, given, HealthCheck, settings, Verbosity

from . import common  # noqa E402
from .common import (
    format_ref_tensors_in_mixed_B_layout,
    gen_mixed_B_batch_sizes,
    MAX_EXAMPLES_LONG_RUNNING,
    open_source,
)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests, TEST_WITH_ROCM
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, optests, TEST_WITH_ROCM

VERBOSITY: Verbosity = Verbosity.verbose

# pyre-ignore
additional_decorators: Dict[str, List[Callable]] = {
    # TODO: Implement the operator registrations later
    "test_faketensor__test_forward_cpu_int8": [
        unittest.skip("Operator not implemented for Meta tensors"),
    ],
    "test_faketensor__test_forward_fused_pooled_emb_quant": [
        unittest.skip("Operator not implemented for Meta tensors"),
    ],
    "test_faketensor__test_forward_gpu_no_cache_int8": [
        unittest.skip("Operator not implemented for Meta tensors"),
    ],
    "test_faketensor__test_forward_gpu_uvm_cache_int8": [
        unittest.skip("Operator not implemented for Meta tensors"),
    ],
}


@optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
class ForwardTest(unittest.TestCase):
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

        if not use_cpu and torch.cuda.is_available():
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
        verbosity=VERBOSITY,
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
        verbosity=VERBOSITY,
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
        verbosity=VERBOSITY,
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
        verbosity=VERBOSITY,
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
        verbosity=VERBOSITY,
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
        verbosity=VERBOSITY,
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


if __name__ == "__main__":
    unittest.main()
