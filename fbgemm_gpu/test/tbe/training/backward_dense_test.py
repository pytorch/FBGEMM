#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import copy
import unittest

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_embedding_utils import (
    b_indices,
    get_table_batched_offsets_from_dense,
    round_up,
    to_device,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    DenseTableBatchedEmbeddingBagsCodegen,
)
from hypothesis import assume, given, HealthCheck, settings, Verbosity

from .. import common  # noqa E402
from ..common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gradcheck, optests, use_cpu_strategy
else:
    from fbgemm_gpu.test.test_utils import gradcheck, optests, use_cpu_strategy


VERBOSITY: Verbosity = Verbosity.verbose


@optests.generate_opcheck_tests(fast=True)
class BackwardDenseTest(unittest.TestCase):
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
        use_cpu=use_cpu_strategy(),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
    )
    @settings(
        verbosity=VERBOSITY,
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
            atol=(
                5.0e-3
                if weights_precision == SparseType.FP16
                or output_dtype == SparseType.FP16
                else 1.0e-5
            ),
            rtol=(
                5.0e-3
                if weights_precision == SparseType.FP16
                or output_dtype == SparseType.FP16
                else 1.0e-5
            ),
        )
        if do_pooling:
            goc = torch.cat([go.view(B, -1) for go in gos], dim=1)
        else:
            goc = torch.cat(gos, dim=0)
        fc2.backward(goc)
        torch.testing.assert_close(
            cc.weights.grad,
            grad_weights,
            atol=(
                5.0e-3
                if weights_precision == SparseType.FP16
                or output_dtype == SparseType.FP16
                else 1.0e-4
            ),
            rtol=(
                5.0e-3
                if weights_precision == SparseType.FP16
                or output_dtype == SparseType.FP16
                else 1.0e-4
            ),
        )

        cc = DenseTableBatchedEmbeddingBagsCodegen(
            [(E, D) for (E, D) in zip(Es, Ds)],
            # NOTE: only SUM pooling can work with per_sample_weights!
            pooling_mode=PoolingMode.SUM,
            use_cpu=use_cpu,
        )

        per_sample_weights = to_device(xw.contiguous().view(-1), use_cpu)
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
        cc = cc.float()
        per_sample_weights = per_sample_weights.float()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        for param in cc.parameters():
            param.requires_grad = False
        gradcheck(
            cc, (indices, offsets, per_sample_weights), eps=1e-2, atol=1e-3, rtol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
