#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Unit tests for fbgemm_gpu.tbe.config.embedding_config

Tests cover:
- Import path correctness
- New classmethod behavior (ComputeDevice.get_available, EmbeddingLocation.from_device_and_clf)
- Utility function correctness
- JIT integration (torch.jit.script must still work on TBE)
"""

import enum
import unittest

import torch

try:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
except ModuleNotFoundError:
    import fbgemm_gpu  # noqa: F401
    from fbgemm_gpu.test.test_utils import gpu_unavailable


class EmbeddingConfigImportTest(unittest.TestCase):
    """Verify all symbols are importable from tbe.config."""

    def test_import_from_tbe_config_package(self) -> None:
        from fbgemm_gpu.tbe.config import (
            BoundsCheckMode,
            ComputeDevice,
            DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
            EmbeddingLocation,
            EmbeddingSpecInfo,
            get_bounds_check_version_for_platform,
            INT8_EMB_ROW_DIM_OFFSET,
            MAX_PREFETCH_DEPTH,
            PoolingMode,
            RecordCacheMetrics,
            round_up,
            SplitState,
            tensor_to_device,
        )

        self.assertTrue(issubclass(EmbeddingLocation, enum.IntEnum))
        self.assertTrue(issubclass(ComputeDevice, enum.IntEnum))
        self.assertTrue(issubclass(PoolingMode, enum.IntEnum))
        self.assertTrue(issubclass(BoundsCheckMode, enum.IntEnum))
        self.assertTrue(issubclass(EmbeddingSpecInfo, enum.IntEnum))
        self.assertTrue(isinstance(DEFAULT_SCALE_BIAS_SIZE_IN_BYTES, int))
        self.assertTrue(callable(get_bounds_check_version_for_platform))
        self.assertTrue(isinstance(INT8_EMB_ROW_DIM_OFFSET, int))
        self.assertTrue(isinstance(MAX_PREFETCH_DEPTH, int))
        self.assertTrue(callable(RecordCacheMetrics))
        self.assertTrue(callable(round_up))
        self.assertTrue(callable(SplitState))
        self.assertTrue(callable(tensor_to_device))


class EmbeddingConfigClassmethodTest(unittest.TestCase):
    """Test new classmethods added to existing enums."""

    def test_compute_device_get_available(self) -> None:
        from fbgemm_gpu.tbe.config import ComputeDevice

        result = ComputeDevice.get_available()
        self.assertIn(result, list(ComputeDevice))

    def test_embedding_location_from_device_and_clf_cpu(self) -> None:
        from fbgemm_gpu.tbe.config import EmbeddingLocation

        self.assertEqual(
            EmbeddingLocation.from_device_and_clf(torch.device("cpu"), 0.0),
            EmbeddingLocation.HOST,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_embedding_location_from_device_and_clf_cuda_managed(self) -> None:
        from fbgemm_gpu.tbe.config import EmbeddingLocation

        device: torch.device = torch.accelerator.current_accelerator()  # pyre-ignore[9]
        assert device is not None
        self.assertEqual(
            EmbeddingLocation.from_device_and_clf(device, 0.0),
            EmbeddingLocation.MANAGED,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_embedding_location_from_device_and_clf_cuda_device(self) -> None:
        from fbgemm_gpu.tbe.config import EmbeddingLocation

        device: torch.device = torch.accelerator.current_accelerator()  # pyre-ignore[9]
        assert device is not None
        self.assertEqual(
            EmbeddingLocation.from_device_and_clf(device, 1.0),
            EmbeddingLocation.DEVICE,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_embedding_location_from_device_and_clf_cuda_caching(self) -> None:
        from fbgemm_gpu.tbe.config import EmbeddingLocation

        device: torch.device = torch.accelerator.current_accelerator()  # pyre-ignore[9]
        assert device is not None
        self.assertEqual(
            EmbeddingLocation.from_device_and_clf(device, 0.5),
            EmbeddingLocation.MANAGED_CACHING,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_embedding_location_from_device_and_clf_invalid_raises(self) -> None:
        from fbgemm_gpu.tbe.config import EmbeddingLocation

        device: torch.device = torch.accelerator.current_accelerator()  # pyre-ignore[9]
        assert device is not None

        with self.assertRaises(ValueError):
            EmbeddingLocation.from_device_and_clf(device, 1.5)
        with self.assertRaises(ValueError):
            EmbeddingLocation.from_device_and_clf(device, -0.1)

    def test_embedding_location_from_str_roundtrip(self) -> None:
        from fbgemm_gpu.tbe.config import EmbeddingLocation

        for loc in EmbeddingLocation:
            self.assertEqual(EmbeddingLocation.from_str(loc.name.lower()), loc)

    def test_pooling_mode_do_pooling(self) -> None:
        from fbgemm_gpu.tbe.config import PoolingMode

        self.assertTrue(PoolingMode.SUM.do_pooling())
        self.assertTrue(PoolingMode.MEAN.do_pooling())
        self.assertFalse(PoolingMode.NONE.do_pooling())


class EmbeddingConfigUtilityTest(unittest.TestCase):
    """Test utility functions."""

    def test_round_up(self) -> None:
        from fbgemm_gpu.tbe.config import round_up

        self.assertEqual(round_up(5, 4), 8)
        self.assertEqual(round_up(4, 4), 4)
        self.assertEqual(round_up(0, 4), 0)
        self.assertEqual(round_up(1, 1), 1)

    def test_tensor_to_device(self) -> None:
        from fbgemm_gpu.tbe.config import tensor_to_device

        t = torch.tensor([1, 2, 3])
        result = tensor_to_device(t, torch.device("cpu"))
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(torch.equal(result, t))

    def test_get_bounds_check_version_for_platform(self) -> None:
        from fbgemm_gpu.tbe.config import get_bounds_check_version_for_platform

        result = get_bounds_check_version_for_platform()
        self.assertIn(result, [1, 2])


class EmbeddingConfigJITTest(unittest.TestCase):
    """Verify types from tbe/config/ don't break torch.jit.script on the TBE module.

    Guards against:
    1. TorchScript failing to resolve IntEnum types at new import paths
    2. Classmethods on IntEnum breaking JIT
    3. Module-level references breaking due to import rewriting
    """

    @unittest.skipIf(*gpu_unavailable)
    def test_tbe_jit_script_with_new_import_paths(self) -> None:
        """Critical: Ensure TBE can still be JIT-scripted."""
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            ComputeDevice,
            EmbeddingLocation,
            SplitTableBatchedEmbeddingBagsCodegen,
        )

        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (100, 16, EmbeddingLocation.DEVICE, ComputeDevice.CUDA),
            ],
        )
        cc_scripted = torch.jit.script(cc)
        indices = torch.randint(
            0, 100, (10,), device=torch.accelerator.current_accelerator()
        )
        offsets = torch.tensor(
            [0, 5, 10], device=torch.accelerator.current_accelerator(), dtype=torch.long
        )
        output = cc_scripted(indices, offsets)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 16)
