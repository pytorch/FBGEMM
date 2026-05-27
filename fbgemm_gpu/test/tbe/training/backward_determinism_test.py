#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

"""Tests for TBE backward determinism.

Verifies that torch.use_deterministic_algorithms(True) produces bit-identical
results across repeated backward passes for all TBE optimizer types, data types,
and pooling modes.
"""

import logging
import os
import unittest
import warnings
from typing import Any, Optional

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.utils import (
    get_table_batched_offsets_from_dense,
    round_up,
    to_device,
)
from hypothesis import assume, given, HealthCheck, settings, Verbosity

from .. import common  # noqa E402
from ..common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import additional_decorators, gpu_unavailable, optests
else:
    from fbgemm_gpu.test.test_utils import (
        additional_decorators,
        gpu_unavailable,
        optests,
    )


VERBOSITY: Verbosity = Verbosity.verbose

SUPPRESS_HEALTH_CHECKS: list[HealthCheck] = [
    HealthCheck.filter_too_much,
    HealthCheck.data_too_large,
    HealthCheck.differing_executors,
]


@optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
class BackwardDeterminismTest(unittest.TestCase):
    """Verify backward determinism for all TBE optimizer types.

    Each test runs the same backward pass multiple times with
    torch.use_deterministic_algorithms(True) and asserts bit-identical
    results (atol=0, rtol=0). The test inputs force long segments
    (duplicate indices in the first column) to exercise the CTA-per-row
    kernel path where the determinism mechanism is critical.

    Result comparison adapts to the optimizer type:
      - dense=True: compares weights.grad tensors
      - optimizer=NONE: compares weights_dev.grad.to_dense() (sparse gradients)
      - All other optimizers: compares split_embedding_weights() per table
    """

    def setUp(self) -> None:
        # The test calls multiple TBE constructors, each emit many debug/info log lines ,
        # Suppress INFO/DEBUG log to reduce the noise.
        # Set FBGEMM_TEST_VERBOSE=1 to re-enable.
        if "FBGEMM_TEST_VERBOSE" not in os.environ:
            warnings.simplefilter("ignore")
            logging.disable(logging.INFO)

    def _run_dense_backward(
        self,
        Es: list[int],
        Ds: list[int],
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        ref_cc: DenseTableBatchedEmbeddingBagsCodegen,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor],
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        cc = DenseTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[(E, D) for (E, D) in zip(Es, Ds)],
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
        )
        cc.weights.data.copy_(ref_cc.weights.data)
        output = cc(indices, offsets, per_sample_weights)
        output.backward(grad_output)
        grad = cc.weights.grad
        assert grad is not None
        return grad.clone()

    def _run_split_backward(
        self,
        T: int,
        Es: list[int],
        Ds: list[int],
        managed: list[EmbeddingLocation],
        optimizer: OptimType,
        weights_precision: SparseType,
        output_dtype: SparseType,
        pooling_mode: PoolingMode,
        extra_kwargs: dict[str, Any],
        ref_cc: SplitTableBatchedEmbeddingBagsCodegen,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor],
        total_unique_indices: Optional[int],
        grad_output: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (E, D, M, ComputeDevice.CUDA) for (E, D, M) in zip(Es, Ds, managed)
            ],
            optimizer=optimizer,
            learning_rate=0.5,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            stochastic_rounding=False,
            pooling_mode=pooling_mode,
            **extra_kwargs,
        )
        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(
                ref_cc.split_embedding_weights()[t].data
            )
        output = cc(
            indices,
            offsets,
            per_sample_weights=per_sample_weights,
            total_unique_indices=total_unique_indices,
        )
        output.backward(grad_output)
        cc.flush()

        if optimizer == OptimType.NONE:
            grad = cc.weights_dev.grad
            assert grad is not None
            # pyre-ignore[29]: Pyre cannot resolve `to_dense` on torch stubs
            return grad.to_dense().clone()
        return [cc.split_embedding_weights()[t].clone() for t in range(T)]

    def _assert_results_deterministic(
        self,
        results: list[Any],
        T: int,
        dense: bool,
        optimizer: OptimType,
    ) -> None:
        for run_idx in range(1, len(results)):
            if dense or optimizer == OptimType.NONE:
                torch.testing.assert_close(
                    results[run_idx],
                    results[0],
                    atol=0,
                    rtol=0,
                    msg=f"Determinism violation: run {run_idx} differs from run 0",
                )
            else:
                for t in range(T):
                    torch.testing.assert_close(
                        results[run_idx][t],
                        results[0][t],
                        atol=0,
                        rtol=0,
                        msg=f"Determinism violation: run {run_idx} table {t} "
                        f"differs from run 0",
                    )

    def _run_backward_determinism(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        optimizer: OptimType,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType = SparseType.FP32,
        dense: bool = False,
        num_runs: int = 5,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Run backward pass num_runs times and assert bit-identical results.

        Args:
            T: Number of embedding tables.
            D: Base embedding dimension (multiplied by 4 internally).
            B: Batch size.
            log_E: Log10 of hash size (number of embedding rows).
            L: Pooling factor (number of indices per sample).
            weighted: Whether to use per-sample weights.
            mixed: Whether to use mixed embedding dimensions across tables.
            optimizer: The TBE optimizer type.
            pooling_mode: SUM, MEAN, or NONE.
            weights_precision: FP16 or FP32 for embedding weights.
            output_dtype: Output tensor dtype (FP16, FP32, or BF16).
            dense: If True, use DenseTableBatchedEmbeddingBagsCodegen.
            num_runs: Number of repeated backward passes to compare.
            optimizer_kwargs: Extra kwargs passed to the TBE constructor
                (e.g., eps, beta1, beta2, ensemble_mode).
        """
        assume(pooling_mode == PoolingMode.SUM or not weighted)
        assume(not mixed or pooling_mode != PoolingMode.NONE)
        assume(not weighted or pooling_mode != PoolingMode.NONE)
        assume(optimizer != OptimType.NONE or not mixed)

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

        managed = [EmbeddingLocation.DEVICE] * T

        rng = np.random.RandomState(42)
        xs = [
            to_device(
                torch.from_numpy(
                    rng.choice(range(e), size=(B, L), replace=True).astype(np.int64)
                ),
                use_cpu=False,
            )
            for e in Es
        ]
        # Force long segments: all rows in each batch share the same index
        # in the first column. This triggers the CTA-per-row kernel path
        # where determinism matters.
        if L > 0:
            for x, e in zip(xs, Es):
                x[:, 0] = rng.randint(low=0, high=e)

        xws = [to_device(torch.randn(size=(B, L)), use_cpu=False) for _ in range(T)]
        xws_flat = torch.cat([xw.contiguous().flatten() for xw in xws], dim=0)

        x = torch.cat([x.contiguous().flatten() for x in xs], dim=0)
        indices, offsets = get_table_batched_offsets_from_dense(
            x, L, T * B, use_cpu=False
        )

        per_sample_weights = (
            to_device(xws_flat.contiguous().view(-1), use_cpu=False)
            if weighted
            else None
        )

        total_unique_indices = None
        if not dense and optimizer == OptimType.NONE:
            total_unique_indices = 0
            for t in range(T):
                start = offsets[t * B]
                end = offsets[(t + 1) * B]
                total_unique_indices += indices[start:end].unique().numel()

        extra_kwargs: dict[str, Any] = optimizer_kwargs or {}

        torch.manual_seed(0)

        # Create reference TBE to get initial weights and output shape
        if dense:
            ref_cc = DenseTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[(E, D) for (E, D) in zip(Es, Ds)],
                pooling_mode=pooling_mode,
                weights_precision=weights_precision,
                output_dtype=output_dtype,
            )
        else:
            ref_cc = SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (E, D, M, ComputeDevice.CUDA) for (E, D, M) in zip(Es, Ds, managed)
                ],
                optimizer=optimizer,
                learning_rate=0.5,
                weights_precision=weights_precision,
                output_dtype=output_dtype,
                stochastic_rounding=False,
                pooling_mode=pooling_mode,
                **extra_kwargs,
            )

        ref_output = (
            ref_cc(indices, offsets, per_sample_weights)
            if dense
            else ref_cc(
                indices,
                offsets,
                per_sample_weights=per_sample_weights,
                total_unique_indices=total_unique_indices,
            )
        )
        grad_output = torch.randn_like(ref_output)

        # Collect results from multiple runs with deterministic mode.
        # use_deterministic_algorithms(True) forces all ops to use deterministic
        # implementations (or raise an error if none exists).
        results = []
        torch.use_deterministic_algorithms(True)
        try:
            for _ in range(num_runs):
                if dense:
                    assert isinstance(ref_cc, DenseTableBatchedEmbeddingBagsCodegen)
                    result = self._run_dense_backward(
                        Es,
                        Ds,
                        pooling_mode,
                        weights_precision,
                        output_dtype,
                        ref_cc,
                        indices,
                        offsets,
                        per_sample_weights,
                        grad_output,
                    )
                else:
                    assert isinstance(ref_cc, SplitTableBatchedEmbeddingBagsCodegen)
                    result = self._run_split_backward(
                        T,
                        Es,
                        Ds,
                        managed,
                        optimizer,
                        weights_precision,
                        output_dtype,
                        pooling_mode,
                        extra_kwargs,
                        ref_cc,
                        indices,
                        offsets,
                        per_sample_weights,
                        total_unique_indices,
                        grad_output,
                    )
                results.append(result)
        finally:
            torch.use_deterministic_algorithms(False)

        self._assert_results_deterministic(results, T, dense, optimizer)

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=64),
        B=st.integers(min_value=2, max_value=32),
        log_E=st.integers(min_value=3, max_value=4),
        L=st.integers(min_value=2, max_value=20),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        pooling_mode=st.sampled_from(
            [PoolingMode.SUM, PoolingMode.MEAN, PoolingMode.NONE]
        ),
        output_dtype=st.sampled_from(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=20,
        deadline=None,
        suppress_health_check=SUPPRESS_HEALTH_CHECKS,
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_determinism_sgd(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        pooling_mode: PoolingMode,
        output_dtype: SparseType,
    ) -> None:
        """Test determinism for EXACT_SGD with FP16/FP32 weights and
        FP16/FP32/BF16 output dtypes."""
        self._run_backward_determinism(
            T=T,
            D=D,
            B=B,
            log_E=log_E,
            L=L,
            weighted=weighted,
            mixed=mixed,
            optimizer=OptimType.EXACT_SGD,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
        )

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=64),
        B=st.integers(min_value=2, max_value=32),
        log_E=st.integers(min_value=3, max_value=4),
        L=st.integers(min_value=2, max_value=20),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        optimizer=st.sampled_from(
            [OptimType.EXACT_ADAGRAD, OptimType.EXACT_ROWWISE_ADAGRAD]
        ),
        pooling_mode=st.sampled_from(
            [PoolingMode.SUM, PoolingMode.MEAN, PoolingMode.NONE]
        ),
        output_dtype=st.sampled_from(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=20,
        deadline=None,
        suppress_health_check=SUPPRESS_HEALTH_CHECKS,
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_determinism_adagrad(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        optimizer: OptimType,
        pooling_mode: PoolingMode,
        output_dtype: SparseType,
    ) -> None:
        """Test determinism for EXACT_ADAGRAD and EXACT_ROWWISE_ADAGRAD with
        FP16/FP32 weights and FP16/FP32/BF16 output dtypes."""
        self._run_backward_determinism(
            T=T,
            D=D,
            B=B,
            log_E=log_E,
            L=L,
            weighted=weighted,
            mixed=mixed,
            optimizer=optimizer,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            optimizer_kwargs={"eps": 1e-4},
        )

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=64),
        B=st.integers(min_value=2, max_value=32),
        log_E=st.integers(min_value=3, max_value=4),
        L=st.integers(min_value=2, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        optimizer=st.sampled_from(
            [
                OptimType.ADAM,
                OptimType.LAMB,
                OptimType.PARTIAL_ROWWISE_LAMB,
                OptimType.LARS_SGD,
                OptimType.EMAINPLACE_ROWWISE_ADAGRAD,
            ]
        ),
        pooling_mode=st.sampled_from(
            [PoolingMode.SUM, PoolingMode.MEAN, PoolingMode.NONE]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=20,
        deadline=None,
        suppress_health_check=SUPPRESS_HEALTH_CHECKS,
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_determinism_optimizers(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        optimizer: OptimType,
        pooling_mode: PoolingMode,
    ) -> None:
        """Test determinism for ADAM, LAMB, PARTIAL_ROWWISE_LAMB,
        LARS_SGD, and EMAINPLACE_ROWWISE_ADAGRAD."""
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            EmainplaceModeDefinition,
        )

        if optimizer == OptimType.LARS_SGD:
            kwargs: dict[str, Any] = {
                "weight_decay": 0.01,
                "momentum": 0.9,
                "eta": 0.01,
            }
        elif optimizer == OptimType.EMAINPLACE_ROWWISE_ADAGRAD:
            D = D * 2
            kwargs = {
                "eps": 1e-4,
                "emainplace_mode": EmainplaceModeDefinition(
                    step_ema=1.0,
                    step_start=0.0,
                    step_ema_coef=0.9,
                ),
            }
        else:
            kwargs = {
                "eps": 1e-4,
                "beta1": 0.9,
                "beta2": 0.99,
                "weight_decay": 0.01,
            }
        self._run_backward_determinism(
            T=T,
            D=D,
            B=B,
            log_E=log_E,
            L=L,
            weighted=weighted,
            mixed=mixed,
            optimizer=optimizer,
            pooling_mode=pooling_mode,
            weights_precision=SparseType.FP32,
            optimizer_kwargs=kwargs,
        )

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=64),
        B=st.integers(min_value=2, max_value=32),
        log_E=st.integers(min_value=3, max_value=4),
        L=st.integers(min_value=2, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        pooling_mode=st.sampled_from(
            [PoolingMode.SUM, PoolingMode.MEAN, PoolingMode.NONE]
        ),
        momentum1_dtype=st.sampled_from([SparseType.FP32, SparseType.BF16]),
        momentum2_dtype=st.sampled_from([SparseType.FP32, SparseType.BF16]),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=20,
        deadline=None,
        suppress_health_check=SUPPRESS_HEALTH_CHECKS,
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_determinism_partial_rowwise_adam(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        pooling_mode: PoolingMode,
        momentum1_dtype: SparseType,
        momentum2_dtype: SparseType,
    ) -> None:
        """Test determinism for PARTIAL_ROWWISE_ADAM with all supported
        optimizer state dtypes for momentum1 and momentum2.
        The codegen instantiates FP32 and BF16 for both
        (see partial_rowwise_adam ph_tys in optimizers.py)."""
        self._run_backward_determinism(
            T=T,
            D=D,
            B=B,
            log_E=log_E,
            L=L,
            weighted=weighted,
            mixed=mixed,
            optimizer=OptimType.PARTIAL_ROWWISE_ADAM,
            pooling_mode=pooling_mode,
            weights_precision=SparseType.FP32,
            optimizer_kwargs={
                "eps": 1e-4,
                "beta1": 0.9,
                "beta2": 0.99,
                "weight_decay": 0.01,
                "optimizer_state_dtypes": {
                    "momentum1": momentum1_dtype,
                    "momentum2": momentum2_dtype,
                },
            },
        )

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=64),
        B=st.integers(min_value=2, max_value=32),
        log_E=st.integers(min_value=3, max_value=4),
        L=st.integers(min_value=2, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        pooling_mode=st.sampled_from(
            [PoolingMode.SUM, PoolingMode.MEAN, PoolingMode.NONE]
        ),
        momentum2_dtype=st.sampled_from([SparseType.FP32, SparseType.BF16]),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=20,
        deadline=None,
        suppress_health_check=SUPPRESS_HEALTH_CHECKS,
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_determinism_ensemble(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        pooling_mode: PoolingMode,
        momentum2_dtype: SparseType,
    ) -> None:
        """Test determinism for ENSEMBLE_ROWWISE_ADAGRAD with all supported
        optimizer state dtypes for momentum2 (momentum1 is always FP32)."""
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            EnsembleModeDefinition,
            StepMode,
        )

        kwargs: dict[str, Any] = {
            "eps": 1e-4,
            "optimizer_state_dtypes": {"momentum2": momentum2_dtype},
            "ensemble_mode": EnsembleModeDefinition(
                step_ema=1.0,
                step_swap=1.0,
                step_start=0.0,
                step_ema_coef=0.9,
                step_mode=StepMode.USE_ITER,
            ),
        }
        self._run_backward_determinism(
            T=T,
            D=D,
            B=B,
            log_E=log_E,
            L=L,
            weighted=weighted,
            mixed=mixed,
            optimizer=OptimType.ENSEMBLE_ROWWISE_ADAGRAD,
            pooling_mode=pooling_mode,
            weights_precision=SparseType.FP32,
            optimizer_kwargs=kwargs,
        )

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=64),
        B=st.integers(min_value=2, max_value=32),
        log_E=st.integers(min_value=3, max_value=4),
        L=st.integers(min_value=2, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        pooling_mode=st.sampled_from(
            [PoolingMode.SUM, PoolingMode.MEAN, PoolingMode.NONE]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=20,
        deadline=None,
        suppress_health_check=SUPPRESS_HEALTH_CHECKS,
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_determinism_none(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        pooling_mode: PoolingMode,
    ) -> None:
        """Test determinism for optimizer=NONE which returns sparse gradients
        instead of updating weights in-place."""
        self._run_backward_determinism(
            T=T,
            D=D,
            B=B,
            log_E=log_E,
            L=L,
            weighted=weighted,
            mixed=mixed,
            optimizer=OptimType.NONE,
            pooling_mode=pooling_mode,
            weights_precision=SparseType.FP32,
        )

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=64),
        B=st.integers(min_value=2, max_value=32),
        log_E=st.integers(min_value=3, max_value=4),
        L=st.integers(min_value=2, max_value=20),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        pooling_mode=st.sampled_from(
            [PoolingMode.SUM, PoolingMode.MEAN, PoolingMode.NONE]
        ),
        output_dtype=st.sampled_from(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=20,
        deadline=None,
        suppress_health_check=SUPPRESS_HEALTH_CHECKS,
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_determinism_dense(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        pooling_mode: PoolingMode,
        output_dtype: SparseType,
    ) -> None:
        """Test determinism for DenseTableBatchedEmbeddingBagsCodegen with
        FP16/FP32 weights and FP16/FP32/BF16 output dtypes."""
        self._run_backward_determinism(
            T=T,
            D=D,
            B=B,
            log_E=log_E,
            L=L,
            weighted=weighted,
            mixed=mixed,
            optimizer=OptimType.NONE,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            dense=True,
        )

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=64),
        B=st.sampled_from([128, 256]),
        L=st.integers(min_value=2, max_value=10),
        optimizer=st.sampled_from(
            [
                OptimType.EXACT_SGD,
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.ADAM,
                OptimType.PARTIAL_ROWWISE_ADAM,
                OptimType.LAMB,
                OptimType.PARTIAL_ROWWISE_LAMB,
                OptimType.LARS_SGD,
            ]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=10,
        deadline=None,
        suppress_health_check=SUPPRESS_HEALTH_CHECKS,
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_determinism_long_segments(
        self,
        T: int,
        D: int,
        B: int,
        L: int,
        optimizer: OptimType,
    ) -> None:
        """Test determinism with large batch sizes (B=128/256) and small hash
        sizes (log_E=2) to create very long segments that exercise the
        CTA-per-row kernel path with segments exceeding
        max_segment_length_per_cta."""
        extra_kwargs: dict[str, Any] = {}
        if optimizer in (
            OptimType.ADAM,
            OptimType.PARTIAL_ROWWISE_ADAM,
            OptimType.LAMB,
            OptimType.PARTIAL_ROWWISE_LAMB,
        ):
            extra_kwargs = {
                "eps": 1e-4,
                "beta1": 0.9,
                "beta2": 0.99,
                "weight_decay": 0.01,
            }
        elif optimizer in (OptimType.EXACT_ADAGRAD, OptimType.EXACT_ROWWISE_ADAGRAD):
            extra_kwargs = {"eps": 1e-4}
        elif optimizer == OptimType.LARS_SGD:
            extra_kwargs = {
                "weight_decay": 0.01,
                "momentum": 0.9,
                "eta": 0.01,
            }
        self._run_backward_determinism(
            T=T,
            D=D,
            B=B,
            log_E=2,
            L=L,
            weighted=False,
            mixed=False,
            optimizer=optimizer,
            pooling_mode=PoolingMode.SUM,
            weights_precision=SparseType.FP32,
            optimizer_kwargs=extra_kwargs,
        )


if __name__ == "__main__":
    unittest.main()
