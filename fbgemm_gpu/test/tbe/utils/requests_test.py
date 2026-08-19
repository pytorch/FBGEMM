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
from fbgemm_gpu import sparse_ops  # noqa: F401
from fbgemm_gpu.tbe.utils import generate_requests_for_grouped_tables


class GroupedTableRequestsTest(unittest.TestCase):
    def _validate_requests(self, alpha: float, weighted: bool) -> None:
        B = 3
        Ls = [2, 0, 3]
        feature_table_map = [0, 1, 0]
        Es = [11, 13]
        with patch(
            "fbgemm_gpu.tbe.utils.requests.torch.cuda.is_available",
            return_value=False,
        ):
            requests = generate_requests_for_grouped_tables(
                2,
                B,
                2,
                max(Ls),
                max(Es),
                Ls=Ls,
                feature_table_map=feature_table_map,
                Es=Es,
                alpha=alpha,
                weighted=weighted,
            )

        self.assertEqual(len(requests), 2)
        expected_lengths = torch.tensor(
            [length for length in Ls for _ in range(B)], dtype=torch.long
        )
        for request in requests:
            indices, offsets, per_sample_weights = request.unpack_3()
            offsets_cpu = offsets.cpu()
            torch.testing.assert_close(
                offsets_cpu[1:] - offsets_cpu[:-1], expected_lengths
            )
            self.assertEqual(indices.numel(), int(expected_lengths.sum().item()))

            for feature, table in enumerate(feature_table_map):
                start = int(offsets_cpu[feature * B].item())
                end = int(offsets_cpu[(feature + 1) * B].item())
                feature_indices = indices[start:end]
                self.assertTrue(torch.all(feature_indices >= 0).item())
                self.assertTrue(torch.all(feature_indices < Es[table]).item())

            if weighted:
                self.assertIsNotNone(per_sample_weights)
                assert per_sample_weights is not None
                self.assertEqual(per_sample_weights.numel(), indices.numel())
            else:
                self.assertIsNone(per_sample_weights)

    def test_uniform_weighted_requests(self) -> None:
        self._validate_requests(alpha=1.0, weighted=True)

    def test_zipf_unweighted_requests(self) -> None:
        self._validate_requests(alpha=1.2, weighted=False)

    def test_rejects_invalid_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "must equal"):
            generate_requests_for_grouped_tables(
                1,
                2,
                2,
                2,
                11,
                Ls=[2, 1],
                feature_table_map=[0],
                Es=[7, 11],
            )

        with self.assertRaisesRegex(ValueError, "must be in"):
            generate_requests_for_grouped_tables(
                1,
                2,
                2,
                2,
                11,
                Ls=[2, 1],
                feature_table_map=[0, 2],
                Es=[7, 11],
            )

        with self.assertRaisesRegex(ValueError, "missing tables"):
            generate_requests_for_grouped_tables(
                1,
                2,
                2,
                2,
                11,
                Ls=[2, 1],
                feature_table_map=[0, 0],
                Es=[7, 11],
            )


if __name__ == "__main__":
    unittest.main()
