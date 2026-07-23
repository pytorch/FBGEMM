#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import logging
import os
import time
import unittest
from typing import Any, Optional

import torch
from fbgemm_gpu.split_embedding_configs import (
    EmbeddingLocation,
    EmbOptimType as OptimType,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
    WeightDecayMode,
)
from fbgemm_gpu.utils import updated_env
from hypothesis import given, settings

from ..common import create_tbe_from_config, load_tbe_configs_from_file  # noqa E402
from .backward_adagrad_common import (
    additional_decorators,
    adjust_mixed_B_st,
    CacheAlgorithm,
    common_settings,
    common_strategy,
    execute_backward_adagrad,
    gpu_memory_lt_gb,
    gpu_unavailable,
    optests,
    PoolingMode,
    skipIfNotRocm,
    SparseType,
    st,
)

# Set up test strategy
test_st: dict[str, Any] = common_strategy.copy()
test_st["D"] = st.integers(min_value=2, max_value=128)
test_st_cpu: dict[str, Any] = test_st.copy()
test_st_cpu["use_cpu"] = st.just(True)
test_st_cpu["row_wise"] = st.just(True)
test_st_cpu["output_dtype"] = st.sampled_from([SparseType.FP32, SparseType.FP16])


@optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
class BackwardAdagradTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(mixed_B=st.booleans(), **test_st)
    @settings(**common_settings)
    def test_backward_adagrad_fp16_pmSUM(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        kwargs = adjust_mixed_B_st(kwargs)
        execute_backward_adagrad(
            weights_precision=SparseType.FP16,
            pooling_mode=PoolingMode.SUM,
            compile=False,  # FIXME: make compilation work for fp16
            **kwargs,
        )

    @optests.dontGenerateOpCheckTests("FP8 compute requires custom op support.")
    @unittest.skipIf(*gpu_unavailable)
    @given(mixed_B=st.booleans(), **test_st)
    @settings(**common_settings)
    def test_backward_adagrad_fp8_pmSUM(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        kwargs = adjust_mixed_B_st(kwargs)
        # Skip for use_cpu=True, as FP8 is not supported on CPU.
        # Also disable on AMD for now.
        if kwargs["use_cpu"] or torch.version.hip:
            return
        execute_backward_adagrad(
            weights_precision=SparseType.NFP8,
            pooling_mode=PoolingMode.SUM,
            compile=False,  # FIXME: make compilation work for fp16
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        mixed_B=st.booleans(),
        compile=st.booleans(),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp16_pmMEAN(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        kwargs = adjust_mixed_B_st(kwargs)
        execute_backward_adagrad(
            weights_precision=SparseType.FP16,
            pooling_mode=PoolingMode.MEAN,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        compile=st.booleans(),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp16_pmNONE(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        execute_backward_adagrad(
            weights_precision=SparseType.FP16,
            pooling_mode=PoolingMode.NONE,
            mixed_B=False,
            **kwargs,
        )

    @given(
        mixed_B=st.booleans(),
        compile=st.booleans(),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp32_pmSUM(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        kwargs = adjust_mixed_B_st(kwargs)
        execute_backward_adagrad(
            weights_precision=SparseType.FP32,
            pooling_mode=PoolingMode.SUM,
            **kwargs,
        )

    @given(
        compile=st.booleans(),
        pooling_mode=st.sampled_from([PoolingMode.SUM, PoolingMode.MEAN]),
        **test_st_cpu,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp32_cpu(  # noqa C901
        self,
        pooling_mode: PoolingMode,
        **kwargs: Any,
    ) -> None:
        """
        Test VBE support for CPU on rowwise adagrad
        """
        kwargs = adjust_mixed_B_st(kwargs)
        execute_backward_adagrad(
            weights_precision=SparseType.FP32,
            pooling_mode=pooling_mode,
            mixed_B=True,
            **kwargs,
        )

    @given(
        compile=st.booleans(),
        **test_st_cpu,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp32_pmNONE_cpu(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        execute_backward_adagrad(
            weights_precision=SparseType.FP32,
            mixed_B=False,
            pooling_mode=PoolingMode.NONE,
            **kwargs,
        )

    @given(
        mixed_B=st.booleans(),
        compile=st.booleans(),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp32_pmMEAN(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        kwargs = adjust_mixed_B_st(kwargs)
        execute_backward_adagrad(
            weights_precision=SparseType.FP32,
            pooling_mode=PoolingMode.MEAN,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        compile=st.booleans(),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp32_pmNONE(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        execute_backward_adagrad(
            weights_precision=SparseType.FP32,
            mixed_B=False,
            pooling_mode=PoolingMode.NONE,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        mixed_B=st.booleans(),
        max_norm=st.floats(min_value=0.01, max_value=1.0),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp16_pmSUM_with_max_norm(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        kwargs = adjust_mixed_B_st(kwargs)
        fixed_strategy = {"row_wise": True, "use_cpu": False}
        for key, val in fixed_strategy.items():
            assert key in kwargs
            kwargs[key] = val
        execute_backward_adagrad(
            weights_precision=SparseType.FP16,
            pooling_mode=PoolingMode.SUM,
            **kwargs,
        )

    def _test_backward_adagrad_rocm_kernel(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        weight_decay_mode: WeightDecayMode,
        output_dtype: Optional[SparseType] = None,
    ) -> None:
        """Helper method for ROCm backward kernel tests."""
        execute_backward_adagrad(
            T=T,
            D=D,
            B=B,
            log_E=log_E,
            L=L,
            D_gradcheck=1,
            weights_precision=weights_precision,
            stochastic_rounding=False,
            weighted=weighted,
            row_wise=True,
            mixed=False,
            mixed_B=False,
            use_cache=False,
            cache_algorithm=CacheAlgorithm.LRU,
            pooling_mode=PoolingMode.SUM,
            use_cpu=False,
            output_dtype=(
                output_dtype if output_dtype is not None else weights_precision
            ),
            weight_decay_mode=weight_decay_mode,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.sampled_from([16, 32, 40, 48, 64, 80]),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=2, max_value=20),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        output_dtype=st.sampled_from(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16]
        ),
        weighted=st.booleans(),
        weight_decay_mode=st.sampled_from(
            [
                WeightDecayMode.NONE,
                WeightDecayMode.L2,
                WeightDecayMode.DECOUPLE,
            ]
        ),
    )
    @settings(**common_settings)
    @unittest.skipIf(*gpu_unavailable)
    @skipIfNotRocm("Test evaluates fallback kernel on ROCm")
    def test_backward_adagrad_rocm_fallback_kernel(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        output_dtype: SparseType,
        weighted: bool,
        weight_decay_mode: WeightDecayMode,
    ) -> None:
        with updated_env(
            {"FBGEMM_NO_JK": "1", "FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL": "0"}
        ):
            logging.info(
                "Testing ROCm backward kernel with FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL=0 (stock)"
            )
            self._test_backward_adagrad_rocm_kernel(
                T=T,
                D=D,
                B=B,
                log_E=log_E,
                L=L,
                weights_precision=weights_precision,
                weighted=weighted,
                weight_decay_mode=weight_decay_mode,
                output_dtype=output_dtype,
            )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.sampled_from([16, 32, 40, 48, 64, 80]),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=2, max_value=20),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        output_dtype=st.sampled_from(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16]
        ),
        weighted=st.booleans(),
        weight_decay_mode=st.sampled_from(
            [
                WeightDecayMode.NONE,
                WeightDecayMode.L2,
                WeightDecayMode.DECOUPLE,
            ]
        ),
    )
    @settings(**common_settings)
    @unittest.skipIf(*gpu_unavailable)
    @skipIfNotRocm("Test evaluates ROCm optimized backward kernel")
    def test_backward_adagrad_rocm_optimized_kernel(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        output_dtype: SparseType,
        weighted: bool,
        weight_decay_mode: WeightDecayMode,
    ) -> None:
        with updated_env(
            {"FBGEMM_NO_JK": "1", "FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL": "1"}
        ):
            logging.info(
                "Testing ROCm backward kernel with FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL=1 (optimized)"
            )
            self._test_backward_adagrad_rocm_kernel(
                T=T,
                D=D,
                B=B,
                log_E=log_E,
                L=L,
                weights_precision=weights_precision,
                weighted=weighted,
                weight_decay_mode=weight_decay_mode,
                output_dtype=output_dtype,
            )

    @given(
        T=st.integers(min_value=2, max_value=4),
        D=st.sampled_from([64, 128, 160, 192, 256, 320]),
        B=st.integers(min_value=2, max_value=32),
        log_E=st.integers(min_value=2, max_value=4),
        L=st.integers(min_value=2, max_value=16),
    )
    @settings(**common_settings)
    @unittest.skipIf(*gpu_unavailable)
    @skipIfNotRocm("Regression test for HIP backward kernel FP16 momentum bug")
    def test_backward_adagrad_rocm_hip_fp16_momentum_regression(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
    ) -> None:
        """
        Regression test for D97502307: validates that the HIP TBE backward
        kernel correctly types momentum as acc_type<cache_t, true> rather
        than cache_t.  When cache_t=half (FP16), the bug causes:
          1. momentum reads/writes as 16-bit instead of 32-bit -> NaN/Inf
          2. wrong pointer stride for tables beyond the first -> corruption

        This test exercises: HIP backward kernel + FP16 cache + rowwise
        AdaGrad + multiple tables, with 4-phase manual validation of
        momentum and weight correctness.
        """
        E = int(10**log_E)
        Es = [E] * T

        lr = 0.5
        eps = 0.1
        weight_decay_mode = WeightDecayMode.NONE

        with updated_env(
            {
                "FBGEMM_NO_JK": "1",
                "FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL": "1",
            }
        ):
            cc = SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)
                    for _ in range(T)
                ],
                weights_precision=SparseType.FP16,
                output_dtype=SparseType.FP16,
                optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
                learning_rate=lr,
                eps=eps,
                weight_decay=0.0,
                weight_decay_mode=weight_decay_mode,
                stochastic_rounding=False,
            )
            cc = cc.cuda()

            initial_weights = [w.float().clone() for w in cc.split_embedding_weights()]

            indices_list = []
            offsets_list = [
                torch.tensor(
                    [0],
                    dtype=torch.long,
                    device=torch.accelerator.current_accelerator(),
                )
            ]
            total = 0
            for t in range(T):
                for _ in range(B):
                    pool_size = L
                    idx = torch.randint(
                        0,
                        Es[t],
                        (pool_size,),
                        device=torch.accelerator.current_accelerator(),
                    )
                    indices_list.append(idx)
                    total += pool_size
                    offsets_list.append(
                        torch.tensor(
                            [total],
                            dtype=torch.long,
                            device=torch.accelerator.current_accelerator(),
                        )
                    )

            indices = torch.cat(indices_list)
            offsets = torch.cat(offsets_list)

            output = cc(indices, offsets)
            self.assertEqual(output.dtype, torch.float16)

            grad_output = torch.ones_like(output)
            output.backward(grad_output)

            # Phase 1 & 2: Validate momentum for all tables
            optimizer_states = cc.split_optimizer_states()
            self.assertEqual(len(optimizer_states), T)

            for t in range(T):
                m1 = optimizer_states[t][0]

                self.assertTrue(
                    torch.all(torch.isfinite(m1)),
                    f"Table {t}: momentum contains NaN or Inf — "
                    f"likely reading half as float. "
                    f"min={m1.min().item()}, max={m1.max().item()}",
                )

                self.assertTrue(
                    torch.all(m1 >= 0),
                    f"Table {t}: momentum contains negative values — "
                    f"corrupted pointer arithmetic. "
                    f"min={m1.min().item()}",
                )

            # Phase 3: Validate weight updates are finite and changed
            updated_weights = [w.float().clone() for w in cc.split_embedding_weights()]
            for t in range(T):
                w_new = updated_weights[t]
                self.assertTrue(
                    torch.all(torch.isfinite(w_new)),
                    f"Table {t}: updated weights contain NaN or Inf",
                )

                accessed_rows = indices_list[t * B : (t + 1) * B]
                accessed_indices = torch.cat(accessed_rows).unique()
                w_old_accessed = initial_weights[t][accessed_indices.cpu()]
                w_new_accessed = w_new[accessed_indices.cpu()]
                self.assertFalse(
                    torch.allclose(w_old_accessed, w_new_accessed, atol=0, rtol=0),
                    f"Table {t}: weights did not change after backward pass",
                )

            # Phase 4: Cross-table momentum consistency check
            m1_means = []
            for t in range(T):
                m1 = optimizer_states[t][0]
                accessed_rows = indices_list[t * B : (t + 1) * B]
                accessed_indices = torch.cat(accessed_rows).unique()
                m1_accessed = m1[accessed_indices.cpu()]
                m1_means.append(m1_accessed.float().mean().item())

            for t in range(1, T):
                ratio = max(m1_means[0], m1_means[t]) / max(
                    min(m1_means[0], m1_means[t]), 1e-10
                )
                self.assertLess(
                    ratio,
                    100.0,
                    f"Table {t} momentum mean ({m1_means[t]:.6f}) is wildly "
                    f"different from table 0 ({m1_means[0]:.6f}) — "
                    f"likely pointer arithmetic bug (ratio={ratio:.1f})",
                )

    @unittest.skipIf(*gpu_unavailable)
    @skipIfNotRocm("Validates stochastic rounding in the optimized HIP backward kernel")
    def test_backward_adagrad_rocm_hip_stochastic_rounding_seeded(self) -> None:
        """
        Validates that the optimized HIP TBE backward kernel threads stochastic
        rounding (SR) correctly. Before SR was plumbed into the HIP optimizer the
        kernel always used round-to-nearest, so the updated weights were
        independent of the RNG seed. With identical weights/inputs and only the
        SR seed varying, we expect:
          - same seed -> bit-identical updated weights (seed threaded correctly)
          - diff seed -> different updated weights (SR is actually applied)
        """
        E: int = 1000
        T, B, L = 1, 64, 10
        D: int = 128
        lr: float = 0.5
        eps: float = 0.1
        device = torch.accelerator.current_accelerator()

        # Fixed weights + inputs so SR is the only source of randomness.
        torch.manual_seed(0)
        init_weights: torch.Tensor = torch.randn(E, D, device=device).to(torch.float16)
        indices: torch.Tensor = torch.randint(
            0, E, (T * B * L,), dtype=torch.long, device=device
        )
        offsets: torch.Tensor = torch.arange(
            0, T * B * L + 1, L, dtype=torch.long, device=device
        )

        def run_update(sr_seed: int) -> torch.Tensor:
            with updated_env(
                {"FBGEMM_NO_JK": "1", "FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL": "1"}
            ):
                cc = SplitTableBatchedEmbeddingBagsCodegen(
                    embedding_specs=[
                        (E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)
                    ],
                    weights_precision=SparseType.FP16,
                    output_dtype=SparseType.FP16,
                    optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
                    learning_rate=lr,
                    eps=eps,
                    weight_decay=0.0,
                    weight_decay_mode=WeightDecayMode.NONE,
                    stochastic_rounding=True,
                ).cuda()
                with torch.no_grad():
                    cc.split_embedding_weights()[0].copy_(init_weights)
                # Seed the default generator that supplies the SR philox state,
                # immediately before the backward so the offset is deterministic.
                torch.manual_seed(sr_seed)
                output = cc(indices, offsets)
                output.backward(torch.ones_like(output))
                return cc.split_embedding_weights()[0].detach().clone()

        w_same_a = run_update(1234)
        w_same_b = run_update(1234)
        w_diff = run_update(5678)

        self.assertTrue(
            torch.equal(w_same_a, w_same_b),
            "Same SR seed produced different weights — the SR seed is not "
            "threaded deterministically through the HIP backward kernel.",
        )
        self.assertFalse(
            torch.equal(w_same_a, w_diff),
            "Different SR seeds produced identical weights — stochastic rounding "
            "is not being applied by the HIP backward kernel (round-to-nearest).",
        )

    @unittest.skipIf(*gpu_unavailable)
    @skipIfNotRocm(
        "Validates stochastic rounding is unbiased in the HIP backward kernel"
    )
    def test_backward_adagrad_rocm_hip_stochastic_rounding_unbiased(self) -> None:
        """
        Stochastic rounding is unbiased: E[SR(x)] == x. Averaging the FP16 SR'd
        weight updates from the optimized HIP kernel over many runs should
        converge to the exact FP32 optimizer update, while any single run does
        not match it (it rounds to the FP16 grid). This is the property that
        recovers NE parity vs. round-to-nearest.
        """
        E: int = 200
        B, L = 32, 8
        D: int = 128
        lr: float = 0.5
        eps: float = 0.1
        n_runs = 50
        device = torch.accelerator.current_accelerator()

        torch.manual_seed(0)
        init_weights_fp16 = torch.randn(E, D, device=device).to(torch.float16)
        indices = torch.randint(0, E, (B * L,), dtype=torch.long, device=device)
        offsets = torch.arange(0, B * L + 1, L, dtype=torch.long, device=device)
        accessed = indices.unique().cpu()

        def build(
            precision: SparseType, stochastic: bool
        ) -> SplitTableBatchedEmbeddingBagsCodegen:
            return SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[(E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)],
                weights_precision=precision,
                output_dtype=precision,
                optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
                learning_rate=lr,
                eps=eps,
                weight_decay=0.0,
                weight_decay_mode=WeightDecayMode.NONE,
                stochastic_rounding=stochastic,
            ).cuda()

        with updated_env(
            {"FBGEMM_NO_JK": "1", "FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL": "1"}
        ):
            # FP32 oracle: exact update with no rounding. The gradient into the
            # table is a scatter of grad_output and is independent of weight
            # precision, so this is the unbiased target for the FP16 SR updates.
            cc_ref = build(SparseType.FP32, False)
            with torch.no_grad():
                cc_ref.split_embedding_weights()[0].copy_(init_weights_fp16.float())
            out = cc_ref(indices, offsets)
            out.backward(torch.ones_like(out))
            w_ref = cc_ref.split_embedding_weights()[0].detach().float().cpu()[accessed]

            # FP16 + SR: average accessed-row updates across many runs.
            acc_sum = torch.zeros_like(w_ref)
            single_run = None
            for r in range(n_runs):
                cc = build(SparseType.FP16, True)
                with torch.no_grad():
                    cc.split_embedding_weights()[0].copy_(init_weights_fp16)
                torch.manual_seed(1000 + r)
                out = cc(indices, offsets)
                out.backward(torch.ones_like(out))
                w_run = cc.split_embedding_weights()[0].detach().float().cpu()[accessed]
                acc_sum += w_run
                if single_run is None:
                    single_run = w_run
            w_sr_mean = acc_sum / n_runs

        # Mean of SR updates converges to the exact FP32 update (unbiased).
        torch.testing.assert_close(w_sr_mean, w_ref, atol=5e-3, rtol=5e-3)
        # A single SR run does not match the exact update bit-for-bit — i.e. SR
        # genuinely rounds to the FP16 grid rather than being a no-op.
        self.assertIsNotNone(single_run)
        self.assertFalse(
            torch.equal(single_run, w_ref),
            "A single SR run exactly matched the FP32 update — SR not applied.",
        )

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(40))
    def test_backward_adagrad_from_config_file(self) -> None:
        """
        Test backward adagrad pass using TBE specs loaded from a JSON file.

        This test reads TBE init parameters from a JSON file, either input_tbe_specs.json by default
        or specified by the TBE_CONFIG_PATH environment variable. It runs backward tests for each
        configuration.

        The config file can be generated using extract_specs_from_log.py:
            python3 ${HOME}/fbsource/fbcode/ai_codesign/nonprod/supadchaya/scripts/fbgemm/extract_specs_from_log.py \
                --specs_log <path_to_specs_log> \
                --planner_log <path_to_planner_log> \
                --output <output_json_path>

        To run this test with a specific config file:
            TBE_CONFIG_PATH=/path/to/config.json \
                buck2 run @//mode/opt fbcode//deeplearning/fbgemm/fbgemm_gpu/test/tbe:backward_adagrad \
                -- -r test_backward_adagrad_from_config_file
        or
            TBE_CONFIG_PATH=/path/to/config.json python3 -m pytest backward_adagrad_test.py \
                -k test_backward_adagrad_from_config_file
        """
        default_config_path = os.path.join(
            os.path.dirname(__file__), "input_tbe_specs.json"
        )
        config_path = os.environ.get("TBE_CONFIG_PATH", default_config_path)
        if not config_path or not os.path.exists(config_path):
            self.skipTest(f"Config file not found: {config_path}")

        batch_size, _, common_config, tbe_configs = load_tbe_configs_from_file(
            config_path
        )

        # Use a smaller batch size for testing to reduce memory usage (otherwise OOM)
        test_batch_size = min(batch_size, 512)
        L = 10 if test_batch_size < batch_size else 2
        config_idx = os.environ.get("TBE_CONFIG_INDEX", None)
        max_config = os.environ.get("TBE_MAX_CONFIG", "5")
        max_config = int(max_config) if max_config != "all" else -1
        configs_to_test = (
            tbe_configs[:max_config]
            if config_idx is None
            else [tbe_configs[int(config_idx)]]
        )

        total_start_time = time.perf_counter()
        end_time = total_start_time

        for i, config in enumerate(configs_to_test):
            with self.subTest(config_index=i):
                start_time = time.perf_counter()
                try:
                    tbe_op = create_tbe_from_config(
                        config, common_config, use_cpu=False
                    )
                    T = len(tbe_op.embedding_specs)
                    max_D = tbe_op.max_D
                    mixed_D = tbe_op.mixed_D
                    execute_backward_adagrad(
                        T=T,
                        D=max_D,
                        B=test_batch_size,
                        log_E=0,
                        L=L,
                        D_gradcheck=1,
                        weights_precision=tbe_op.weights_precision,
                        stochastic_rounding=tbe_op.stochastic_rounding,
                        weighted=False,
                        row_wise=(tbe_op.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD),
                        mixed=mixed_D,
                        mixed_B=False,
                        use_cache=False,
                        cache_algorithm=tbe_op.cache_algorithm,
                        pooling_mode=tbe_op.pooling_mode,
                        use_cpu=False,
                        output_dtype=SparseType.from_int(tbe_op.output_dtype),
                        weight_decay_mode=tbe_op.weight_decay_mode,
                        tbe_op=tbe_op,
                    )
                    end_time = time.perf_counter()
                    logging.info(
                        f"TBE_DEBUG: PASSED for config {i} T={T} max_D={max_D} (mixed: {mixed_D}), the test took {(end_time - start_time) / 60:.2f} mins"
                    )
                except Exception as e:
                    # Log but don't fail - some configs may have unsupported features
                    end_time = time.perf_counter()
                    raise RuntimeError(
                        f"TEST FAILED for config {i} uuid={config.get('tbe_uuid', '')}, the test took {(end_time - start_time) / 60:.2f} mins and failed: {e}"
                    )
        logging.info(
            f"TBE_DEBUG: Total time taken to test {len(configs_to_test)} configs is {(end_time - total_start_time) / 60:.2f} mins"
        )

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(20))
    def test_backward_adagrad_with_simple_tbe_op(self) -> None:
        """
        Test execute_backward_adagrad with a pre-created TBE op.

        This test verifies that execute_backward_adagrad works correctly
        with a manually created TBE op passed via the cc parameter.
        """
        T = 57  # Number of tables
        D = 4  # Embedding dimension
        E = 10  # Number of embeddings per table
        B = 294400  # Batch size
        L = 1  # Pooling factor
        pooling_mode = PoolingMode.SUM
        weights_precision = SparseType.FP32
        output_dtype = SparseType.FP32
        embedding_specs = [
            (E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA) for _ in range(T)
        ]

        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            weights_precision=weights_precision,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            learning_rate=0.05,
            eps=0.2,
            pooling_mode=pooling_mode,
            output_dtype=output_dtype,
        )

        execute_backward_adagrad(
            T=T,
            D=D,
            B=B,
            log_E=1,
            L=L,
            D_gradcheck=1,
            weights_precision=weights_precision,
            stochastic_rounding=False,
            weighted=False,
            row_wise=True,
            mixed=False,
            mixed_B=False,
            use_cache=False,
            cache_algorithm=CacheAlgorithm.LRU,
            pooling_mode=pooling_mode,
            use_cpu=False,
            output_dtype=output_dtype,
            tbe_op=cc,
        )


if __name__ == "__main__":
    unittest.main()
