#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import unittest

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

from .. import common  # noqa E402
from ..common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


class WeightInitDeviceTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_split_tbe_weight_init_device_cpu(self) -> None:
        T = 3
        E = 100
        D = 32
        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA) for _ in range(T)
            ],
            weights_precision=SparseType.FP32,
            optimizer=OptimType.NONE,
            pooling_mode=PoolingMode.SUM,
            weight_init_device=torch.device("cpu"),
        )
        # weights_dev should be on CPU
        # pyre-ignore[29]: weights_dev is a Tensor
        self.assertEqual(cc.weights_dev.device.type, "cpu")
        # pyre-ignore[29]: weights_dev is a Tensor
        self.assertEqual(cc.weights_dev.numel(), T * E * D)
        # metadata should remain on CUDA
        self.assertEqual(cc.weights_offsets.device.type, "cuda")
        self.assertEqual(cc.weights_placements.device.type, "cuda")

    @unittest.skipIf(*gpu_unavailable)
    def test_split_tbe_weight_init_device_none(self) -> None:
        T = 2
        E = 64
        D = 16
        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA) for _ in range(T)
            ],
            weights_precision=SparseType.FP32,
            optimizer=OptimType.NONE,
            pooling_mode=PoolingMode.SUM,
            weight_init_device=None,
        )
        # default: weights_dev should be on CUDA
        self.assertEqual(cc.weights_dev.device.type, "cuda")
        self.assertEqual(cc.weights_offsets.device.type, "cuda")

    @unittest.skipIf(*gpu_unavailable)
    def test_split_tbe_weight_init_device_does_not_affect_optimizer_states(
        self,
    ) -> None:
        T = 2
        E = 64
        D = 16
        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA) for _ in range(T)
            ],
            weights_precision=SparseType.FP32,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            learning_rate=0.01,
            pooling_mode=PoolingMode.SUM,
            weight_init_device=torch.device("cpu"),
        )
        # weights_dev on CPU
        self.assertEqual(cc.weights_dev.device.type, "cpu")
        # optimizer state (momentum1) should remain on CUDA
        self.assertEqual(cc.momentum1_dev.device.type, "cuda")

    @unittest.skipIf(*gpu_unavailable)
    def test_dense_tbe_weight_init_device_cpu(self) -> None:
        E = 100
        D = 32
        T = 3
        cc = DenseTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[(E, D) for _ in range(T)],
            pooling_mode=PoolingMode.SUM,
            weight_init_device=torch.device("cpu"),
        )
        # weights should be on CPU
        self.assertEqual(cc.weights.device.type, "cpu")
        self.assertEqual(cc.weights.numel(), T * E * D)
        # metadata should remain on CUDA
        self.assertEqual(cc.D_offsets.device.type, "cuda")
        self.assertEqual(cc.hash_size_cumsum.device.type, "cuda")

    @unittest.skipIf(*gpu_unavailable)
    def test_dense_tbe_weight_init_device_none(self) -> None:
        E = 64
        D = 16
        T = 2
        cc = DenseTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[(E, D) for _ in range(T)],
            pooling_mode=PoolingMode.SUM,
            weight_init_device=None,
        )
        # default: weights on CUDA
        self.assertEqual(cc.weights.device.type, "cuda")

    @unittest.skipIf(*gpu_unavailable)
    def test_split_tbe_weight_init_device_cpu_move_to_cuda(self) -> None:
        T = 2
        E = 64
        D = 16
        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA) for _ in range(T)
            ],
            weights_precision=SparseType.FP32,
            optimizer=OptimType.NONE,
            pooling_mode=PoolingMode.SUM,
            weight_init_device=torch.device("cpu"),
        )
        self.assertEqual(cc.weights_dev.device.type, "cpu")
        # Simulate user moving weights to CUDA after init
        # pyre-ignore[16, 6]: weights_dev is a Tensor
        cc.weights_dev = torch.nn.Parameter(cc.weights_dev.data.cuda())
        self.assertEqual(cc.weights_dev.device.type, "cuda")

    @unittest.skipIf(*gpu_unavailable)
    def test_dense_tbe_weight_init_device_cpu_move_to_cuda(self) -> None:
        E = 64
        D = 16
        T = 2
        cc = DenseTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[(E, D) for _ in range(T)],
            pooling_mode=PoolingMode.SUM,
            weight_init_device=torch.device("cpu"),
        )
        self.assertEqual(cc.weights.device.type, "cpu")
        # Simulate user moving weights to CUDA after init
        cc.weights = torch.nn.Parameter(cc.weights.cuda())
        self.assertEqual(cc.weights.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
