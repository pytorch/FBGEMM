#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.config.embedding_config import (
    ComputeDevice,
    EmbeddingLocation,
    PoolingMode,
)
from hypothesis import given, settings, Verbosity

from ..common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


VERBOSITY: Verbosity = Verbosity.verbose

# Two tables, small enough that the expected forward output can be written out
# by hand below.
T = 2
E = 8
D = 4


def _reference_rows() -> torch.Tensor:
    """Distinct value per (row, dim) so a misplaced row shows up in the output."""
    return torch.arange(T * E * D, dtype=torch.float32).reshape(T * E, D)


def _as_tensor(value: object) -> torch.Tensor:
    """Narrow a registered-buffer lookup, which is typed `Tensor | Module | int`."""
    assert isinstance(value, torch.Tensor)
    return value


class WeightInitOnCpuTest(unittest.TestCase):
    def _make_split_tbe(
        self,
        weight_init_on_cpu: bool,
        optimizer: OptimType = OptimType.EXACT_ROWWISE_ADAGRAD,
        location: EmbeddingLocation = EmbeddingLocation.DEVICE,
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        return SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[(E, D, location, ComputeDevice.CUDA) for _ in range(T)],
            optimizer=optimizer,
            learning_rate=0.1,
            weights_precision=SparseType.FP32,
            output_dtype=SparseType.FP32,
            pooling_mode=PoolingMode.SUM,
            weight_init_on_cpu=weight_init_on_cpu,
        )

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    @given(weight_init_on_cpu=st.booleans())
    @settings(verbosity=VERBOSITY, max_examples=2, deadline=None)
    def test_split_tbe_weight_placement(self, weight_init_on_cpu: bool) -> None:
        cc = self._make_split_tbe(weight_init_on_cpu=weight_init_on_cpu)
        init_device = "cpu" if weight_init_on_cpu else "cuda"

        self.assertEqual(cc.weight_init_on_cpu, weight_init_on_cpu)
        self.assertEqual(cc.weights_dev.device.type, init_device)
        self.assertEqual(_as_tensor(cc.weights_dev).numel(), T * E * D)

        # Everything else stays on the compute device either way. The offsets and
        # placements are dereferenced by the kernels, and optimizer state is
        # outside the flag's scope entirely.
        self.assertEqual(cc.weights_offsets.device.type, "cuda")
        self.assertEqual(cc.weights_placements.device.type, "cuda")
        self.assertEqual(cc.D_offsets.device.type, "cuda")
        self.assertEqual(cc.momentum1_dev.device.type, "cuda")

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_split_tbe_uvm_tables_are_unaffected(self) -> None:
        # With every table on UVM the dev buffer is empty, so redirecting it buys
        # nothing; it stays on the compute device and the TBE is usable as-is.
        cc = self._make_split_tbe(
            weight_init_on_cpu=True,
            location=EmbeddingLocation.MANAGED,
        )

        self.assertEqual(_as_tensor(cc.weights_dev).numel(), 0)
        self.assertEqual(cc.weights_dev.device.type, "cuda")

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_split_tbe_forward_matches_after_move_to_compute_device(self) -> None:
        cc = self._make_split_tbe(weight_init_on_cpu=True)
        self.assertEqual(cc.weights_dev.device.type, "cpu")

        # Populate the tables while they are still on the host -- the point of the
        # feature is that the weights are filled in (loaded from a checkpoint)
        # before they ever reach HBM.
        rows = _reference_rows()
        for t, table in enumerate(cc.split_embedding_weights()):
            table.copy_(rows[t * E : (t + 1) * E])

        cc.to(torch.accelerator.current_accelerator())
        self.assertEqual(cc.weights_dev.device.type, "cuda")

        # B = 2, one index per bag. Bags are feature-major: bags 0..1 belong to
        # table 0, bags 2..3 to table 1.
        indices = torch.tensor(
            [1, 3, 0, 6],
            dtype=torch.int64,
            device=torch.accelerator.current_accelerator(),
        )
        offsets = torch.tensor(
            [0, 1, 2, 3, 4],
            dtype=torch.int64,
            device=torch.accelerator.current_accelerator(),
        )
        output = cc(indices, offsets)

        expected = torch.stack(
            [
                torch.cat([rows[1], rows[E + 0]]),
                torch.cat([rows[3], rows[E + 6]]),
            ]
        ).to(torch.accelerator.current_accelerator())
        torch.testing.assert_close(output, expected)

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_split_tbe_forward_before_move_raises(self) -> None:
        # Documents the failure mode promised by the `weight_init_on_cpu` docs: a
        # forward that skips the move fails in the op rather than reading garbage.
        #
        # The check is `TENSOR_ON_GPU_OR_MTIA(weights[0])` in
        # codegen/training/pt2/embedding_split_host_pt2_autograd_template.cpp,
        # which runs in the host wrapper before any kernel launch. `weights[0]`
        # is `weights_dev` -- the one buffer this flag redirects -- so the
        # message names the offending tensor and the device it is stuck on.
        cc = self._make_split_tbe(weight_init_on_cpu=True)
        indices = torch.tensor(
            [1, 3, 0, 6],
            dtype=torch.int64,
            device=torch.accelerator.current_accelerator(),
        )
        offsets = torch.tensor(
            [0, 1, 2, 3, 4],
            dtype=torch.int64,
            device=torch.accelerator.current_accelerator(),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"weights\[0\] must be a GPU or MTIA tensor; "
            r"it is currently on device CPU",
        ):
            cc(indices, offsets)
