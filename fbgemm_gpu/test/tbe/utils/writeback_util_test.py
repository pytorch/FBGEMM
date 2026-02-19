# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from fbgemm_gpu.utils.writeback_util import writeback_update_gradient_ec
from later.unittest import TestCase


class WritebackUtilsTest(TestCase):
    def test_writeback_update_gradient_ec_empty_indices(self) -> None:
        """Test that empty indices returns first element of grad unchanged."""
        indices = torch.tensor([], dtype=torch.long)
        offsets = torch.tensor([0, 0], dtype=torch.long)
        grad = (torch.tensor([1.0, 2.0, 3.0]),)
        feature_table_map = [0]

        result = writeback_update_gradient_ec(indices, offsets, grad, feature_table_map)

        self.assertEqual(result.size(), grad[0].size())
        self.assertTrue(torch.equal(result, grad[0]))

    def test_writeback_update_gradient_ec_no_duplicates(self) -> None:
        """Test case where all indices are unique - all gradients should be preserved."""
        # Single table, batch size 2, 2 indices per sample (pooling factor = 1 for EC)
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        offsets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        grad = (torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),)
        feature_table_map = [0]

        result = writeback_update_gradient_ec(indices, offsets, grad, feature_table_map)

        # All indices are unique, so all gradients should be preserved
        expected = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        self.assertTrue(torch.equal(result, expected))

    def test_writeback_update_gradient_ec_with_duplicates(self) -> None:
        """Test that only the first occurrence of each index gets the gradient."""
        # Single table, indices with duplicates
        # Indices: [0, 1, 0, 2] - index 0 appears at positions 0 and 2
        indices = torch.tensor([0, 1, 0, 2], dtype=torch.long)
        offsets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        grad = (torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),)
        feature_table_map = [0]

        result = writeback_update_gradient_ec(indices, offsets, grad, feature_table_map)

        # Position 0 has index 0 (first occurrence) -> keeps gradient
        # Position 1 has index 1 (first occurrence) -> keeps gradient
        # Position 2 has index 0 (duplicate) -> should be masked out
        # Position 3 has index 2 (first occurrence) -> keeps gradient
        expected = torch.tensor([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [4.0, 4.0]])
        self.assertEqual(result.size(), expected.size())
        self.assertTrue(torch.equal(result, expected))

    def test_writeback_update_gradient_ec_multiple_tables(self) -> None:
        """Test that duplicates across different tables are NOT deduped."""
        # Two tables, batch size 2
        # Table 0: indices [0, 1], Table 1: indices [0, 1]
        # Same index in different tables should NOT be deduped
        indices = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.long)
        offsets = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
        grad = (torch.tensor([10.0, 20.0, 20.0, 30.0, 40.0, 40.0]),)
        feature_table_map = [0, 1]

        result = writeback_update_gradient_ec(indices, offsets, grad, feature_table_map)

        # batch_size = 6 / 2 = 4
        # Table 0 has indices at offsets 0, 1, 2 (batch 0, 1)
        # Table 1 has indices at offsets 2, 3, 4 (batch 0, 1)
        # Index 0 in table 0 != index 0 in table 1 due to table offset
        # All gradients should be preserved since there are no duplicates within same table except the last one in each batch.
        expected = torch.tensor([10.0, 20.0, 0, 30.0, 40.0, 0])
        self.assertEqual(result.size(), expected.size())
        self.assertTrue(torch.equal(result, expected))

    def test_writeback_update_gradient_ec_all_same_index(self) -> None:
        """Test case where all indices are the same - only first should get gradient."""
        indices = torch.tensor([5, 5, 5, 5], dtype=torch.long)
        offsets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        grad = (torch.tensor([[1.0], [2.0], [3.0], [4.0]]),)
        feature_table_map = [0]

        result = writeback_update_gradient_ec(indices, offsets, grad, feature_table_map)

        # Only the first occurrence (position 0) should keep its gradient
        expected = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
        self.assertEqual(result.size(), expected.size())
        self.assertTrue(torch.equal(result, expected))
