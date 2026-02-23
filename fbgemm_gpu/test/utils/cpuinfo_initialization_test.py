#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Minimal test to detect cpuinfo initialization bug between PyTorch and FBGEMM.

WHY THIS TEST USES SplitEmbInferenceConverter:
----------------------------------------------
The cpuinfo bug causes corruption in FBGEMM's *index remapping* logic during
the inference converter's pruning step. Simple quantization ops (like
FloatToFused8BitRowwiseQuantized) produce identical results regardless of
which ISA (scalar/AVX2/AVX512) is used - they're mathematically equivalent.

The bug ONLY manifests when:
1. Multiple embedding tables are used (T >= 2)
2. Pruning is applied (which creates index remapping tables)
3. The corrupted cpuinfo causes wrong JIT kernel generation for index remapping

This results in 50% of output elements being wrong (one table's index
remapping is corrupted).

To create a truly "minimal" test without TBE, we would need to:
- Directly call FBGEMM's C++ index remapping functions
- Or expose fbgemmInstructionSet() to Python and check its return value

For now, this is the simplest test that reliably detects the bug.

How to run:
    buck2 test //deeplearning/fbgemm/fbgemm_gpu/test/utils:cpuinfo_initialization_test

Expected:
- With good PyTorch (PR #174927 applied): PASS
- With bad PyTorch (commit c5aa299b04...): FAIL with ~50% element mismatch
"""

import unittest

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_embedding_inference_converter import SplitEmbInferenceConverter
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)


class SparseArch(torch.nn.Module):
    """Simple wrapper module for SplitTableBatchedEmbeddingBagsCodegen."""

    def __init__(
        self,
        emb_dim: int,
        num_tables: int,
        num_rows: int,
    ) -> None:
        super().__init__()
        self.emb_module = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (num_rows, emb_dim, EmbeddingLocation.HOST, ComputeDevice.CPU)
                for _ in range(num_tables)
            ],
            weights_precision=SparseType.FP32,
            optimizer=OptimType.EXACT_SGD,
            learning_rate=0.05,
            pooling_mode=PoolingMode.SUM,
        )

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.emb_module(indices, offsets)


class CpuinfoInitializationTest(unittest.TestCase):
    """
    Test for cpuinfo initialization bug that causes index remapping corruption.

    This is the MINIMAL reproduction of test_l2_norm_pruning_workflow.
    """

    def test_pruning_index_remapping(self) -> None:
        """
        Minimal reproduction of the cpuinfo bug.

        Key parameters (from the failing test):
        - D = 128 (embedding dimension)
        - T = 2 (number of tables - MUST be > 1 to see 50% corruption)
        - E = 5 (rows per table)
        - pruning_ratio = 0.5

        With cpuinfo bug: ~50% of elements mismatch (one table corrupted)
        Without bug: All elements match within tolerance
        """
        D = 128
        T = 2
        E = 5

        # Create simple indices and offsets for 4 batches
        indices = torch.tensor([3, 0, 2, 2, 3, 4, 2], dtype=torch.int32)
        offsets = torch.tensor([0, 1, 4, 6, 7], dtype=torch.int32)

        # Create embedding weights with distinct values per row
        weights = [
            (torch.tensor([0.4, 0.1, -0.2, 0.2, 0.3]).view(E, 1)) * torch.ones(E, D),
            (torch.tensor([-0.8, 0.2, 0.5, -0.1, 0.9]).view(E, 1)) * torch.ones(E, D),
        ]

        # For pruning_ratio=0.5, these are the expected remapped indices
        remapped_indices = torch.tensor([3, 0, 2, 2, 4, 2], dtype=torch.int32)
        remapped_offsets = torch.tensor([0, 1, 4, 5, 6], dtype=torch.int32)

        # Create model
        model = SparseArch(emb_dim=D, num_tables=T, num_rows=E)

        # Set weights
        for idx in range(T):
            model.emb_module.split_embedding_weights()[idx].copy_(weights[idx])

        # Get reference output (using remapped indices, before conversion)
        ref_output = model(remapped_indices, remapped_offsets)

        # Apply pruning + quantization conversion
        # THIS IS WHERE CPUINFO IS USED - if cpuinfo is broken,
        # the index remapping for one or more tables gets corrupted
        converter = SplitEmbInferenceConverter(
            quantize_type=SparseType.FP16,
            pruning_ratio=0.5,
            use_array_for_index_remapping=False,
        )
        converter.convert_model(model)

        # Get output after conversion (using original indices)
        converted_output = model(indices, offsets)

        # Compare - with cpuinfo bug, expect ~50% mismatch at table boundary
        try:
            torch.testing.assert_close(
                ref_output.float(),
                converted_output.float(),
                atol=1e-1,
                rtol=1e-1,
            )
        except AssertionError as e:
            # Provide detailed error message
            diff = (ref_output - converted_output).abs()
            mismatch = (diff > 0.1).sum().item()
            total = diff.numel()
            raise AssertionError(
                f"cpuinfo bug detected!\n"
                f"Mismatched elements: {mismatch} / {total} ({100*mismatch/total:.1f}%)\n"
                f"If ~50% mismatch, this confirms index remapping corruption.\n"
                f"Original error: {e}"
            ) from e


if __name__ == "__main__":
    unittest.main(verbosity=2)
