# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, Set, Tuple

import hypothesis.strategies as st
import torch
from fbgemm_gpu.tbe.utils import get_table_batched_offsets_from_dense
from fbgemm_gpu.utils.writeback_util import (
    writeback_gradient,
    writeback_update_gradient_nobag,
)
from hypothesis import given


class WritebackUtilsTest(unittest.TestCase):
    def test_writeback_update_gradient_nobag_empty_indices(self) -> None:
        """Test that empty indices returns first element of grad unchanged."""
        # No indices to look up — simulates a batch with no embedding accesses
        indices = torch.tensor([], dtype=torch.long)
        # Offsets reflect an empty lookup: both start and end point to 0
        offsets = torch.tensor([0, 0], dtype=torch.long)
        # Gradient tensor with 3 rows, embedding dimension 1
        grad = (torch.tensor([[1.0], [2.0], [3.0]]),)
        # Single feature mapped to table 0
        feature_table_map = [0]

        result = writeback_update_gradient_nobag(
            indices, offsets, grad, feature_table_map
        )

        # With no indices, the gradient should pass through unmodified
        self.assertEqual(result.size(), grad[0].size())
        self.assertTrue(torch.equal(result, grad[0]))

    def test_writeback_update_gradient_nobag_no_duplicates(self) -> None:
        """Test case where all indices are unique - all gradients should be preserved."""
        # Single table, batch size 4, each sample looks up exactly 1 unique index
        # Since nobag mode uses pooling factor = 1, each offset interval contains 1 index
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        # 5 offsets for 4 samples: offsets[i] to offsets[i+1] gives index range per sample
        offsets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        # Gradient has 4 rows (one per index) with embedding dimension 2
        grad = (torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),)
        # One feature mapped to table 0
        feature_table_map = [0]

        result = writeback_update_gradient_nobag(
            indices, offsets, grad, feature_table_map
        )

        # All indices are unique within the table, so no deduplication occurs —
        # every gradient row should be preserved as-is
        expected = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        self.assertTrue(torch.equal(result, expected))

    def test_writeback_update_gradient_nobag_with_duplicates(self) -> None:
        """Test that only the first occurrence of each index gets the gradient."""
        # Single table with duplicate indices: index 0 appears at positions 0 and 2
        # This simulates the scenario where multiple samples in a batch reference
        # the same embedding row, requiring deduplication for correct gradient updates
        indices = torch.tensor([0, 1, 0, 2], dtype=torch.long)
        offsets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        # 4 gradient rows (one per lookup), embedding dimension 2
        grad = (torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),)
        feature_table_map = [0]

        result = writeback_update_gradient_nobag(
            indices, offsets, grad, feature_table_map
        )

        # Deduplication logic keeps only the first occurrence of each index:
        # Position 0: index 0 (first occurrence) -> gradient preserved [1.0, 1.0]
        # Position 1: index 1 (first occurrence) -> gradient preserved [2.0, 2.0]
        # Position 2: index 0 (duplicate)        -> gradient zeroed out [0.0, 0.0]
        # Position 3: index 2 (first occurrence) -> gradient preserved [4.0, 4.0]
        expected = torch.tensor([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [4.0, 4.0]])
        self.assertEqual(result.size(), expected.size())
        self.assertTrue(torch.equal(result, expected))

    def test_writeback_update_gradient_nobag_multiple_tables(self) -> None:
        """Test that duplicates across different tables are NOT deduped."""
        # Two tables with one feature each, batch size 3
        # Indices are laid out contiguously: [table0_sample0, table0_sample1,
        #   table0_sample2, table1_sample0, table1_sample1, table1_sample2]
        # Index 1 appears in both tables, but deduplication is per-table,
        # so cross-table duplicates should be kept
        indices = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.long)
        # 7 offsets for 6 lookups (2 tables * 3 samples)
        offsets = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
        # 1-D gradients (embedding dimension 1, squeezed)
        grad = (torch.tensor([10.0, 20.0, 20.0, 30.0, 40.0, 40.0]),)
        # Two features, each mapped to its own table
        feature_table_map = [0, 1]

        result = writeback_update_gradient_nobag(
            indices, offsets, grad, feature_table_map
        )

        # batch_size = total_indices / num_features = 6 / 2 = 3
        # Table 0 indices [0, 1, 1]: index 1 is duplicated at positions 1 and 2
        #   -> position 0 (index 0, first): keep 10.0
        #   -> position 1 (index 1, first): keep 20.0
        #   -> position 2 (index 1, dup):   zero out
        # Table 1 indices [0, 1, 1]: index 1 is duplicated at positions 4 and 5
        #   -> position 3 (index 0, first): keep 30.0
        #   -> position 4 (index 1, first): keep 40.0
        #   -> position 5 (index 1, dup):   zero out
        # Note: index 0 and index 1 appearing in both tables are independent —
        # deduplication scopes are isolated per table
        expected = torch.tensor([10.0, 20.0, 0, 30.0, 40.0, 0])
        self.assertEqual(result.size(), expected.size())
        self.assertTrue(torch.equal(result, expected))

    def test_writeback_update_gradient_nobag_all_same_index(self) -> None:
        """Test case where all indices are the same in one table with one feature - only first should get gradient."""
        # Extreme deduplication scenario: every sample in the batch references
        # the same embedding row (index 5). Only the first occurrence should
        # retain its gradient; all subsequent ones should be zeroed out.
        indices = torch.tensor([5, 5, 5, 5], dtype=torch.long)
        offsets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        # Gradient with 4 rows, embedding dimension 2
        grad = (torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1]]),)
        feature_table_map = [0]

        result = writeback_update_gradient_nobag(
            indices, offsets, grad, feature_table_map
        )

        # Only position 0 (first occurrence of index 5) keeps its gradient;
        # positions 1, 2, 3 are all duplicates and get zeroed out
        expected = torch.tensor(
            [[1.0000, 1.1000], [0.0000, 0.0000], [0.0000, 0.0000], [0.0000, 0.0000]]
        )
        self.assertEqual(result.size(), expected.size())
        self.assertTrue(torch.equal(result, expected))

    def test_writeback_update_gradient_nobag_multiple_features(self) -> None:
        """One table with multiple features — deduplication happens within each feature's
        index range independently, not across the entire table."""
        # Single table backing two features, batch size 3
        # Feature 0 indices: [5, 5, 5] — all the same, so only the first is kept
        # Feature 1 indices: [4, 4, 4] — all the same, so only the first is kept
        indices = torch.tensor([5, 5, 5, 4, 4, 4], dtype=torch.long)
        # 7 offsets for 6 lookups (2 features * 3 samples)
        offsets = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
        # 1-D gradients (embedding dimension 1, squeezed)
        grad = (torch.tensor([10.0, 20.0, 20.0, 30.0, 40.0, 40.0]),)
        # Both features share table 0 (common when a single embedding table
        # backs multiple features)
        feature_table_map = [0]

        result = writeback_update_gradient_nobag(
            indices, offsets, grad, feature_table_map
        )

        # Feature 0: positions 0-2 all have index 5
        #   -> position 0 (first): keep 10.0
        #   -> positions 1-2 (dups): zero out
        # Feature 1: positions 3-5 all have index 4
        #   -> position 3 (first): keep 30.0
        #   -> positions 4-5 (dups): zero out
        expected = torch.tensor([10.0, 0, 0, 30.0, 0, 0])
        self.assertEqual(result.size(), expected.size())
        self.assertTrue(torch.equal(result, expected))

    # Pyre-ignore [56]: Pyre was not able to infer the type of argument
    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=16, max_value=20),
    )
    def test_writeback_update_gradient_nobag(
        self,
        T: int,  # number of tables
        D: int,  # embedding dimension
        B: int,  # batch size
    ) -> None:
        """Property-based test using Hypothesis to verify deduplication across
        random configurations of tables, embedding dimensions, and batch sizes.
        Validates the writeback_gradient wrapper (which calls writeback_update_gradient_nobag
        internally) against a reference implementation."""
        L = 1  # nobag mode: exactly one lookup per sample per feature
        E = 100  # number of embedding rows per table

        # Generate random index tensors for each table, shape (B, L)
        # Each sample looks up one random embedding row from [0, E)
        xs = [torch.randint(low=0, high=E, size=(B, L)) for _ in range(T)]

        # Stack into a dense (T, B, L) tensor and convert to the flat
        # (indices, offsets) representation used by TBE kernels
        x = torch.cat([xi.view(1, B, L) for xi in xs], dim=0)
        indices, offsets = get_table_batched_offsets_from_dense(x, use_cpu=True)

        # Random gradient tensor matching the total number of lookups (T * B)
        # with the specified embedding dimension D
        grad = (torch.rand((indices.size(0), D)),)
        # Identity feature-to-table mapping: feature i -> table i
        feature_table_map = list(range(T))

        # Call the higher-level writeback_gradient API with nobag=True
        result = writeback_gradient(
            grad,
            indices,
            offsets,
            feature_table_map,
            writeback_first_feature_only=False,
            nobag=True,
        )

        # Verify output shape matches input gradient shape
        self.assertEqual(result[0].size(), grad[0].size())

        # Reference implementation: for each table, scan indices in order and
        # only keep the gradient for the first occurrence of each (table, index)
        # pair. Duplicate occurrences get zeroed out.
        expected = torch.zeros_like(grad[0], device=grad[0].device)
        seen: Set[Tuple[int, int]] = set()
        for i in range(indices.size(0)):
            # Determine which table this index belongs to based on position
            table_idx = i // B
            key = (table_idx, cast(int, indices[i].item()))
            if key not in seen:
                seen.add(key)
                expected[i] = grad[0][i]
            # else: leave as zero (duplicate within the same table)
        self.assertTrue(torch.equal(result[0], expected))
