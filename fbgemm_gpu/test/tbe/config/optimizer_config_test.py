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

import enum
import unittest
from typing import Final

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
        self.assertEqual(CounterBasedRegularizationDefinition().counter_halflife, -1)
        self.assertEqual(CowClipDefinition().weight_norm_coefficient, 0.0)
        self.assertEqual(GlobalWeightDecayDefinition().start_iter, 0)
        self.assertFalse(UserEnabledConfigDefinition().use_rowwise_bias_correction)
        self.assertEqual(EnsembleModeDefinition().step_ema, 10000)
        self.assertEqual(EmainplaceModeDefinition().step_ema, 10)
        self.assertEqual(TailIdThreshold().val, 0)


class OptimizerConfigJITTest(unittest.TestCase):
    """Verify optimizer types don't interfere with TBE JIT scripting."""

    @unittest.skipIf(CUDA_NOT_AVAILABLE, "CUDA required")
    def test_tbe_jit_script_with_weight_decay_l2(self) -> None:
        from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            ComputeDevice,
            EmbeddingLocation,
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
        from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            ComputeDevice,
            CounterBasedRegularizationDefinition,
            EmbeddingLocation,
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
                counter_halflife=100,
            ),
        )
        cc_scripted = torch.jit.script(cc)
        device = torch.accelerator.current_accelerator()
        indices = torch.randint(0, 100, (10,), device=device)
        offsets = torch.tensor([0, 5, 10], device=device, dtype=torch.long)
        output = cc_scripted(indices, offsets)
        self.assertEqual(output.shape, (2, 16))
