#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import copy
import unittest

import fbgemm_gpu
import hypothesis.strategies as st
import numpy as np
import torch

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_embedding_utils import (
    b_indices,
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

from . import common  # noqa E402,F401
from .common import (  # noqa E402
    format_ref_tensors_in_mixed_B_layout,
    gen_mixed_B_batch_sizes,
    MAX_EXAMPLES,
    MAX_EXAMPLES_LONG_RUNNING,
)

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_unavailable, optests, TEST_WITH_ROCM
else:
    from fbgemm_gpu.test.test_utils import (
        gpu_available,
        gpu_unavailable,
        optests,
        TEST_WITH_ROCM,
    )


VERBOSITY: Verbosity = Verbosity.verbose


@optests.generate_opcheck_tests(fast=True)
class BackwardSGDTest(unittest.TestCase):
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
        mixed_B=st.booleans(),
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
        verbosity=VERBOSITY,
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
        mixed_B: bool,
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
            mixed_B if not use_cpu else False,
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
        mixed_B=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
    )
    @settings(
        verbosity=VERBOSITY,
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
        mixed_B: bool,
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
            mixed_B,
            use_cache,
            cache_algorithm,
            True,  # long_segments
            PoolingMode.SUM,  # pooling_mode
            False,  # use_cpu
            SparseType.FP32,  # output_dtype
        )


if __name__ == "__main__":
    unittest.main()
