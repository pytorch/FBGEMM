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
    WeightDecayMode,
)

from hypothesis import assume, given, HealthCheck, settings, Verbosity

from . import common  # noqa E402,F401
from .common import (  # noqa E402
    format_ref_tensors_in_mixed_B_layout,
    gen_mixed_B_batch_sizes,
    MAX_EXAMPLES_LONG_RUNNING,
)

torch.ops.import_module("fbgemm_gpu.sparse_ops")

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import (
        gpu_available,
        gpu_unavailable,
        gradcheck,
        optests,
        TEST_WITH_ROCM,
    )
else:
    from fbgemm_gpu.test.test_utils import (
        gpu_available,
        gpu_unavailable,
        gradcheck,
        optests,
        TEST_WITH_ROCM,
    )


VERBOSITY: Verbosity = Verbosity.verbose


@optests.generate_opcheck_tests(fast=True)
class BackwardAdagradTest(unittest.TestCase):
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
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        for param in cc.parameters():
            param.requires_grad = False
        gradcheck(
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

        if gpu_available and not TEST_WITH_ROCM:
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

    @unittest.skipIf(*gpu_unavailable)
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
        verbosity=VERBOSITY,
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

    @unittest.skipIf(*gpu_unavailable)
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
        verbosity=VERBOSITY,
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

    @unittest.skipIf(*gpu_unavailable)
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
        verbosity=VERBOSITY,
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
        verbosity=VERBOSITY,
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
        verbosity=VERBOSITY,
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

    @unittest.skipIf(*gpu_unavailable)
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
        verbosity=VERBOSITY,
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


if __name__ == "__main__":
    unittest.main()
