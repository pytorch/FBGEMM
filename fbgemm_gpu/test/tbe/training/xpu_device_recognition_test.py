#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import patch

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    DenseTableBatchedEmbeddingBagsCodegen,
    get_available_compute_device,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.config.embedding_config import (
    ComputeDevice,
    EmbeddingLocation,
    PoolingMode,
)


class XPUDeviceRecognitionTest(unittest.TestCase):
    def test_get_available_compute_device_selects_xpu(self) -> None:
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.mtia, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=True),
        ):
            self.assertEqual(get_available_compute_device(), ComputeDevice.XPU)

    def test_split_xpu_uses_accelerator_schema(self) -> None:
        module = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (4, 4, EmbeddingLocation.DEVICE, ComputeDevice.XPU),
            ],
            optimizer=EmbOptimType.EXACT_SGD,
            pooling_mode=PoolingMode.NONE,
            device=torch.device("meta"),
        )

        self.assertFalse(module.use_cpu)
        self.assertEqual(module.current_device.type, "meta")

    def test_split_xpu_default_device_does_not_query_cuda(self) -> None:
        real_torch_device = torch.device

        def redirect_xpu_to_meta(device: object) -> torch.device:
            resolved = real_torch_device(device)
            return real_torch_device("meta") if resolved.type == "xpu" else resolved

        with (
            patch.object(torch.xpu, "current_device", return_value=3) as xpu_current,
            patch.object(
                torch.cuda,
                "current_device",
                side_effect=AssertionError("CUDA current_device must not be called"),
            ) as cuda_current,
            patch.object(torch, "device", side_effect=redirect_xpu_to_meta),
        ):
            module = SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (4, 4, EmbeddingLocation.DEVICE, ComputeDevice.XPU),
                ],
                optimizer=EmbOptimType.EXACT_SGD,
                pooling_mode=PoolingMode.NONE,
            )

        self.assertEqual(module.current_device.type, "meta")
        xpu_current.assert_called_once_with()
        cuda_current.assert_not_called()

    def test_dense_xpu_device_bypasses_cuda(self) -> None:
        real_torch_device = torch.device

        def redirect_xpu_to_cpu(device: object) -> torch.device:
            resolved = real_torch_device(device)
            return real_torch_device("cpu") if resolved.type == "xpu" else resolved

        with (
            patch.object(
                torch.cuda,
                "current_device",
                side_effect=AssertionError("CUDA current_device must not be called"),
            ) as cuda_current,
            patch.object(torch, "device", side_effect=redirect_xpu_to_cpu),
        ):
            module = DenseTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[(4, 4)],
                pooling_mode=PoolingMode.NONE,
                device="xpu",
            )

        self.assertFalse(module.use_cpu)
        self.assertFalse(module.use_mtia)
        self.assertEqual(module.current_device.type, "cpu")
        cuda_current.assert_not_called()


if __name__ == "__main__":
    unittest.main()
