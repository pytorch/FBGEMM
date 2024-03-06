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
from typing import Any, Callable, Dict, List, Optional, Union

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_embedding_optimizer_ops import (
    SplitEmbeddingArgs,
    SplitEmbeddingOptimizerParams,
    SplitEmbeddingRowwiseAdagrad,
)
from fbgemm_gpu.split_embedding_utils import (
    b_indices,
    get_table_batched_offsets_from_dense,
    round_up,
    to_device,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from hypothesis import assume, given, HealthCheck, settings, Verbosity
from torch import Tensor

from .. import common  # noqa E402
from ..common import MAX_EXAMPLES, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests, TEST_WITH_ROCM
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, optests, TEST_WITH_ROCM
VERBOSITY: Verbosity = Verbosity.verbose

# pyre-ignore
additional_decorators: Dict[str, List[Callable]] = {
    "test_schema__test_backward_none_with_rowwise_adagrad": [
        unittest.skip("Cannot access data pointer of Tensor that doesn't have storage")
    ],
    "test_faketensor__test_backward_none_with_rowwise_adagrad": [
        unittest.skip("Cannot access data pointer of Tensor that doesn't have storage")
    ],
    "test_autograd_registration__test_backward_none_with_rowwise_adagrad": [
        unittest.skip("Cannot access data pointer of Tensor that doesn't have storage")
    ],
}


@optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
class BackwardNoneTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
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
        output_dtype=st.sampled_from(
            [SparseType.FP16, SparseType.FP32, SparseType.BF16]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_none(self, **kwargs: Any) -> None:
        self.execute_backward_none_(**kwargs)

    @unittest.skipIf(*gpu_unavailable)
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
        output_dtype=st.sampled_from(
            [SparseType.FP16, SparseType.FP32, SparseType.BF16]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
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
        # TODO: Check why long_segments=True fails when output_dtype ==
        # SparseType.BF16
        assume(not long_segments or output_dtype != SparseType.BF16)

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

        if weights_precision == SparseType.FP16:
            xws = [xw.half() for xw in xws]

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws], dim=0)

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
            # Torch's Embedding only produces an output that has the same type
            # as weight
            if weights_precision != output_dtype:
                fs = [f.to(output_dtype.as_dtype()) for f in fs]
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
            assert type(gos) is list
            if do_pooling:
                goc = torch.cat([go.view(B, -1) for go in gos], dim=1)
            else:
                goc = torch.cat(gos, dim=0)
        else:
            assert type(gos) is Tensor
            goc = gos.clone()
        fc2.backward(goc)

        if optimizer is not None:
            params = SplitEmbeddingOptimizerParams(weights_dev=cc.weights_dev)
            embedding_args = SplitEmbeddingArgs(
                weights_placements=cc.weights_placements,
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


if __name__ == "__main__":
    unittest.main()
