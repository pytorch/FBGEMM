# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from fbgemm_gpu.tbe.ssd import DramKvPerfStat, SSDTableBatchedEmbeddingBags


class DramKVPerfStatsParsingTest(unittest.TestCase):
    """Tests for DRAM KV perf stats dictionary parsing."""

    def test_full_vector_yields_every_enum_key(self) -> None:
        """A vector matching the enum length yields a dict keyed by every member."""
        members = list(DramKvPerfStat.__members__.values())
        raw_stats = [float(i) for i in range(len(members))]
        result = SSDTableBatchedEmbeddingBags.parse_dram_kv_perf_stats(raw_stats)

        self.assertEqual(set(result.keys()), set(members))

    def test_keys_have_no_duplicates(self) -> None:
        """Enum string values must be unique so the parsed dict has no collisions."""
        values = [member.value for member in DramKvPerfStat.__members__.values()]
        self.assertEqual(len(values), len(set(values)))

    def test_parsed_dict_values_match_input_order(self) -> None:
        """Each enum member maps to the value at its declared position."""
        members = list(DramKvPerfStat.__members__.values())
        raw_stats = [float(i * 10) for i in range(len(members))]
        result = SSDTableBatchedEmbeddingBags.parse_dram_kv_perf_stats(raw_stats)

        for i, member in enumerate(members):
            self.assertEqual(
                result[member],
                float(i * 10),
                f"Member {member!r} at index {i} has wrong value",
            )

    def test_short_vector_omits_trailing_keys(self) -> None:
        """A vector shorter than the enum yields a dict missing the trailing keys.

        This is the contract `_report_dram_kv_perf_stats` relies on when guarding
        optional metrics with `key in stats` checks.
        """
        members = list(DramKvPerfStat.__members__.values())
        truncated_len = len(members) - 2
        raw_stats = [1.0] * truncated_len

        result = SSDTableBatchedEmbeddingBags.parse_dram_kv_perf_stats(raw_stats)

        self.assertEqual(len(result), truncated_len)
        for member in members[:truncated_len]:
            self.assertIn(member, result)
        for member in members[truncated_len:]:
            self.assertNotIn(member, result)
