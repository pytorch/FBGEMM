#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Unit tests for fbgemm_gpu.tbe.config.optimizer_config

Tests cover:
- Import path correctness
- JIT integration (TBE with various optimizer configs must still JIT-script)
"""

import dataclasses
import enum
import unittest
from typing import Any, Final

import torch

CUDA_NOT_AVAILABLE: Final[bool] = not torch.cuda.is_available()


class OptimizerConfigImportTest(unittest.TestCase):
    """Verify all optimizer types importable from tbe.config."""

    def test_import_all_optimizer_types(self) -> None:
        from fbgemm_gpu.tbe.config import (
            CounterBasedRegularizationDefinition,
            CounterWeightDecayMode,
            CowClipDefinition,
            DoesNotHavePrefix,
            EmainplaceModeDefinition,
            EnsembleModeDefinition,
            GlobalWeightDecayDefinition,
            GradSumDecay,
            LearningRateMode,
            StepMode,
            TailIdThreshold,
            UserEnabledConfigDefinition,
            WeightDecayMode,
        )

        self.assertTrue(issubclass(WeightDecayMode, enum.IntEnum))
        self.assertTrue(issubclass(CounterWeightDecayMode, enum.IntEnum))
        self.assertTrue(issubclass(StepMode, enum.IntEnum))
        self.assertTrue(issubclass(LearningRateMode, enum.IntEnum))
        self.assertTrue(issubclass(GradSumDecay, enum.IntEnum))
        self.assertTrue(issubclass(DoesNotHavePrefix, Exception))
        self.assertEqual(LearningRateMode.PER_ROW_WARMUP, 3)
        self.assertEqual(CounterBasedRegularizationDefinition().counter_halflife, -1)
        self.assertEqual(CounterBasedRegularizationDefinition().warmup_steps, 0)
        self.assertEqual(CounterBasedRegularizationDefinition().warmup_start_value, 0.0)
        self.assertEqual(CowClipDefinition().weight_norm_coefficient, 0.0)
        self.assertEqual(GlobalWeightDecayDefinition().start_iter, 0)
        self.assertFalse(UserEnabledConfigDefinition().use_rowwise_bias_correction)
        self.assertEqual(EnsembleModeDefinition().step_ema, 10000)
        self.assertEqual(EmainplaceModeDefinition().step_ema, 10)
        self.assertEqual(TailIdThreshold().val, 0)


class OptimizerConfigValidationTest(unittest.TestCase):
    def test_per_row_warmup_rejects_incompatible_counter_decay(self) -> None:
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            ComputeDevice,
            CounterBasedRegularizationDefinition,
            CounterWeightDecayMode,
            EmbeddingLocation,
            LearningRateMode,
            OptimType,
            SplitTableBatchedEmbeddingBagsCodegen,
            WeightDecayMode,
        )

        invalid_settings = (
            (CounterWeightDecayMode.L2, -1),
            (CounterWeightDecayMode.DECOUPLE, -1),
        )
        for counter_weight_decay_mode, counter_halflife in invalid_settings:
            with self.subTest(
                counter_weight_decay_mode=counter_weight_decay_mode,
                counter_halflife=counter_halflife,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "PER_ROW_WARMUP.*counter_weight_decay_mode",
                ):
                    SplitTableBatchedEmbeddingBagsCodegen(
                        embedding_specs=[
                            (4, 4, EmbeddingLocation.HOST, ComputeDevice.CPU),
                        ],
                        optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
                        weight_decay_mode=WeightDecayMode.COUNTER,
                        counter_based_regularization=CounterBasedRegularizationDefinition(
                            counter_weight_decay_mode=counter_weight_decay_mode,
                            counter_halflife=counter_halflife,
                            learning_rate_mode=LearningRateMode.PER_ROW_WARMUP,
                            warmup_steps=10,
                        ),
                    )

    def test_per_row_warmup_accepts_adagradw(self) -> None:
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            ComputeDevice,
            CounterBasedRegularizationDefinition,
            CounterWeightDecayMode,
            EmbeddingLocation,
            LearningRateMode,
            OptimType,
            SplitTableBatchedEmbeddingBagsCodegen,
            WeightDecayMode,
        )

        for counter_halflife in (-1, -2):
            with self.subTest(counter_halflife=counter_halflife):
                tbe = SplitTableBatchedEmbeddingBagsCodegen(
                    embedding_specs=[
                        (4, 4, EmbeddingLocation.HOST, ComputeDevice.CPU),
                    ],
                    optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
                    weight_decay=0.1,
                    weight_decay_mode=WeightDecayMode.COUNTER,
                    counter_based_regularization=CounterBasedRegularizationDefinition(
                        counter_weight_decay_mode=CounterWeightDecayMode.ADAGRADW,
                        counter_halflife=counter_halflife,
                        learning_rate_mode=LearningRateMode.PER_ROW_WARMUP,
                        warmup_steps=10,
                    ),
                )
                self.assertEqual(tbe.optimizer_args.warmup_steps, 10)

    def test_counter_config_without_warmup_fields(self) -> None:
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            ComputeDevice,
            CounterBasedRegularizationDefinition,
            EmbeddingLocation,
            OptimType,
            SplitTableBatchedEmbeddingBagsCodegen,
            WeightDecayMode,
        )

        # A model packaged before the warmup fields existed unpickles its config
        # with the packaged class, which has no warmup attributes.
        packaged_fields: list[tuple[str, Any, Any]] = []
        for f in dataclasses.fields(CounterBasedRegularizationDefinition):
            if f.name in ("warmup_steps", "warmup_start_value"):
                continue
            if f.default_factory is not dataclasses.MISSING:
                default = dataclasses.field(default_factory=f.default_factory)
            else:
                default = f.default
            packaged_fields.append((f.name, f.type, default))
        packaged_definition = dataclasses.make_dataclass(
            "CounterBasedRegularizationDefinition", packaged_fields, frozen=True
        )
        tbe = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (4, 4, EmbeddingLocation.HOST, ComputeDevice.CPU),
            ],
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            weight_decay_mode=WeightDecayMode.COUNTER,
            counter_based_regularization=packaged_definition(),
        )
        self.assertEqual(tbe.optimizer_args.warmup_steps, 0)
        self.assertEqual(tbe.optimizer_args.warmup_start_value, 0.0)


class OptimizerConfigJITTest(unittest.TestCase):
    """Verify optimizer types don't interfere with TBE JIT scripting."""

    @unittest.skipIf(CUDA_NOT_AVAILABLE, "CUDA required")
    def test_tbe_jit_script_with_weight_decay_l2(self) -> None:
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            ComputeDevice,
            EmbeddingLocation,
            OptimType,
            SplitTableBatchedEmbeddingBagsCodegen,
            WeightDecayMode,
        )

        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (100, 16, EmbeddingLocation.DEVICE, ComputeDevice.CUDA),
            ],
            optimizer=OptimType.EXACT_SGD,
            learning_rate=0.01,
            weight_decay_mode=WeightDecayMode.L2,
            weight_decay=0.001,
        )
        cc_scripted = torch.jit.script(cc)
        device = torch.accelerator.current_accelerator()
        indices = torch.randint(0, 100, (10,), device=device)
        offsets = torch.tensor([0, 5, 10], device=device, dtype=torch.long)
        output = cc_scripted(indices, offsets)
        self.assertEqual(output.shape, (2, 16))

    @unittest.skipIf(CUDA_NOT_AVAILABLE, "CUDA required")
    def test_tbe_jit_script_with_counter_regularization(self) -> None:
        """Counter-based regularization exercises the most complex dataclass path."""
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            ComputeDevice,
            CounterBasedRegularizationDefinition,
            EmbeddingLocation,
            LearningRateMode,
            OptimType,
            SplitTableBatchedEmbeddingBagsCodegen,
            WeightDecayMode,
        )

        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (100, 16, EmbeddingLocation.DEVICE, ComputeDevice.CUDA),
            ],
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            learning_rate=0.01,
            weight_decay_mode=WeightDecayMode.COUNTER,
            counter_based_regularization=CounterBasedRegularizationDefinition(
                counter_halflife=-1,
                learning_rate_mode=LearningRateMode.PER_ROW_WARMUP,
                warmup_steps=100,
                warmup_start_value=0.2,
            ),
        )
        cc_scripted = torch.jit.script(cc)
        device = torch.accelerator.current_accelerator()
        indices = torch.randint(0, 100, (10,), device=device)
        offsets = torch.tensor([0, 5, 10], device=device, dtype=torch.long)
        output = cc_scripted(indices, offsets)
        self.assertEqual(output.shape, (2, 16))
