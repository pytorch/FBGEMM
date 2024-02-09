#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import copy
import math
import unittest
from typing import Any, Dict, Optional, Tuple, Union

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
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
    CounterBasedRegularizationDefinition,
    CounterWeightDecayMode,
    CowClipDefinition,
    GradSumDecay,
    LearningRateMode,
    SplitTableBatchedEmbeddingBagsCodegen,
    TailIdThreshold,
    WeightDecayMode,
)
from hypothesis import assume, given, HealthCheck, settings, Verbosity

from .. import common  # noqa E402
from ..common import (
    format_ref_tensors_in_mixed_B_layout,
    gen_mixed_B_batch_sizes,
    MAX_EXAMPLES_LONG_RUNNING,
    open_source,
)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests, TEST_WITH_ROCM, use_cpu_strategy
else:
    from fbgemm_gpu.test.test_utils import (
        gpu_unavailable,
        optests,
        TEST_WITH_ROCM,
        use_cpu_strategy,
    )


VERBOSITY: Verbosity = Verbosity.verbose


@optests.generate_opcheck_tests(fast=True)
class BackwardOptimizersTest(unittest.TestCase):
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
        weight_decay_mode: WeightDecayMode = WeightDecayMode.NONE,
        uvm_non_rowwise_momentum: bool = False,
    ) -> None:
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        assume(not use_cpu or T * B * L * D <= 2048)
        assume(
            not use_cpu
            or optimizer
            in [
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_SGD,
            ]
            or (
                optimizer in [OptimType.EXACT_ROWWISE_ADAGRAD]
                and weight_decay_mode
                not in [WeightDecayMode.COUNTER, WeightDecayMode.COWCLIP]
            )
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

        optimizer_kwargs: Dict[str, Any] = {"learning_rate": 0.5}
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
        cowclip_regularization: CowClipDefinition

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
                # fmt: off
                optimizer_kwargs["counter_based_regularization"] = (
                    counter_based_regularization
                )
                # fmt: on

            if weight_decay_mode == WeightDecayMode.COWCLIP:
                cowclip_regularization = CowClipDefinition(
                    counter_weight_decay_mode=CounterWeightDecayMode.DECOUPLE,
                    counter_halflife=10,
                    weight_norm_coefficient=0.5,
                    lower_bound=1e-6,
                )
                optimizer_kwargs["cowclip_regularization"] = cowclip_regularization

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

                if rowwise and weight_decay_mode in (
                    WeightDecayMode.COUNTER,
                    WeightDecayMode.COWCLIP,
                ):
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
                    elif weight_decay_mode in (
                        WeightDecayMode.COUNTER,
                        WeightDecayMode.COWCLIP,
                    ):
                        iter_ = int(cc.iter.item())
                        (
                            dense_cpu_grad,
                            row_counter,
                            freq,
                        ) = self._get_grad_from_counter_adagrad(
                            dense_cpu_grad,
                            bs[t].weight.cpu(),
                            (
                                counter_based_regularization
                                if weight_decay_mode == WeightDecayMode.COUNTER
                                else cowclip_regularization
                            ),
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
                    elif weight_decay_mode == WeightDecayMode.COUNTER:
                        max_counter = cc.max_counter.item()
                        weights_ref = self._get_wts_from_counter_adagrad_using_counter(
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
                    elif weight_decay_mode == WeightDecayMode.COWCLIP:
                        weights_ref = self._get_wts_from_counter_adagrad_using_cowclip(
                            dense_cpu_grad,
                            bs[t].weight.cpu(),
                            denom,
                            cowclip_regularization,
                            row_counter,
                            lr,
                            weight_decay,
                        )
                    else:  # WeightDecayMode.L2 or WeightDecayMode.NONE
                        # pyre-fixme[58]: `/` is not supported for operand types `float`
                        #  and `Tensor`.
                        weights_ref = bs[t].weight.cpu() - lr * dense_cpu_grad / denom
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
                if rowwise and weight_decay_mode in (
                    WeightDecayMode.COUNTER,
                    WeightDecayMode.COWCLIP,
                ):
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

    def _get_grad_from_counter_adagrad(
        self,
        dense_cpu_grad: torch.Tensor,
        weights: torch.Tensor,
        regularization: Union[CounterBasedRegularizationDefinition, CowClipDefinition],
        row_counter: torch.Tensor,
        prev_iter: torch.Tensor,
        iter_: int,
        weight_decay: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_counter = row_counter.view(row_counter.numel(), 1)
        prev_iter = prev_iter.view(prev_iter.numel(), 1)
        freq = torch.ones_like(row_counter)
        counter_weight_decay_mode = regularization.counter_weight_decay_mode
        counter_halflife = regularization.counter_halflife
        l2_wd = 1.0 if counter_weight_decay_mode == CounterWeightDecayMode.L2 else 0.0

        if counter_halflife > 0:
            freq = torch.tensor([counter_halflife]) / row_counter

        if isinstance(regularization, CounterBasedRegularizationDefinition):
            dense_cpu_grad += l2_wd * freq * weight_decay * weights
        else:
            dense_cpu_grad += l2_wd * weight_decay * weights
        return dense_cpu_grad, row_counter, freq

    def _get_wts_from_counter_adagrad_using_counter(
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

    def _get_wts_from_counter_adagrad_using_cowclip(
        self,
        dense_cpu_grad: torch.Tensor,
        weights: torch.Tensor,
        denom: torch.Tensor,
        regularization: CowClipDefinition,
        row_counter: torch.Tensor,
        learning_rate: float,
        weight_decay: float,
    ) -> torch.Tensor:
        counter_weight_decay_mode = regularization.counter_weight_decay_mode
        weight_norm_coefficient = regularization.weight_norm_coefficient
        lower_bound = regularization.lower_bound

        multiplier = torch.tensor([learning_rate]) / denom
        exp_reg_correction = 1.0

        weight_norm = weights.norm(dim=-1, keepdim=True) * weight_norm_coefficient
        clip_ratio = (
            row_counter
            * torch.maximum(weight_norm, torch.Tensor([lower_bound]))
            / dense_cpu_grad.norm(dim=-1, keepdim=True)
        )
        adjusted_multiplier = (
            torch.minimum(clip_ratio, torch.Tensor([1.0])) * multiplier
        )

        if counter_weight_decay_mode == CounterWeightDecayMode.DECOUPLE:
            exp_reg_correction = 1.0 - weight_decay * learning_rate

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
        use_cpu=use_cpu_strategy(),
        uvm_non_rowwise_momentum=st.booleans(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skip(
        "is flaky, see https://www.internalfb.com/intern/test/281475047227145?ref_report_id=0"
    )
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
        use_cpu=use_cpu_strategy(),
        weight_decay_mode=st.sampled_from(
            [
                WeightDecayMode.NONE,
                WeightDecayMode.L2,
                WeightDecayMode.DECOUPLE,
                WeightDecayMode.COUNTER,
                WeightDecayMode.COWCLIP,
            ]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
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
        use_cpu=use_cpu_strategy(),
    )
    @settings(
        verbosity=VERBOSITY,
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
        use_cpu=use_cpu_strategy(),
    )
    @settings(
        verbosity=VERBOSITY,
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


if __name__ == "__main__":
    unittest.main()
