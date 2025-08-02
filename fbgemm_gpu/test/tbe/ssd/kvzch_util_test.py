# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import time
import unittest

import numpy as np
import torch

from fbgemm_gpu.kvzch_util import get_kv_zch_eviction_mask, parse_metadata_tensor
from torchrec.modules.embedding_configs import (
    CountBasedEvictionPolicy,
    CountTimestampMixedEvictionPolicy,
    NoEvictionPolicy,
    TimestampBasedEvictionPolicy,
    VirtualTableEvictionPolicy,
)

from ..common import gpu_unavailable


@unittest.skipIf(*gpu_unavailable)
class KvzchUtilsTest(unittest.TestCase):
    def test_basic_parsing(self) -> None:
        """
        Test typical parsing including used=0 and used=1 cases.
        """
        # Compose metadata values as 64-bit integers:
        # [timestamp=7, count=13, used=0]
        v1 = 7 | (13 << 32)  # used=0 (highest bit not set)
        # [timestamp=42, count=99, used=1]
        # Used=1 is highest bit; encode as a negative int64 in Python to avoid overflow
        v2 = (42 | (99 << 32)) - (1 << 63)  # set highest bit
        # [timestamp=0xABCDEF01, count=0x1ABCDE0, used=1]
        v3 = (0xABCDEF01 | (0x1ABCDE0 << 32)) - (1 << 63)
        vals = [v1, v2, v3]
        tensor = torch.tensor(vals, dtype=torch.int64)

        timestamps, counts, used = parse_metadata_tensor(tensor)

        np.testing.assert_array_equal(
            timestamps.numpy(), np.array([7, 42, 0xABCDEF01], dtype=np.uint32)
        )
        np.testing.assert_array_equal(
            counts.numpy(), np.array([13, 99, 0x1ABCDE0], dtype=np.uint32)
        )
        np.testing.assert_array_equal(
            used.numpy(), np.array([False, True, True], dtype=bool)
        )

    def test_edge_cases(self) -> None:
        """
        Test edge cases including all zeros, max values, min values, and different used flags.
        """
        # All fields zero, used=0
        v1 = 0
        # Max timestamp, max count, used=0
        v2 = 0xFFFFFFFF | (0x7FFFFFFF << 32)  # Used=0 (highest bit = 0)
        # Min timestamp, min count, used=1
        v3 = 0 - (1 << 63)  # All fields 0, only highest bit set (used=1)

        vals = [v1, v2, v3]
        tensor = torch.tensor(vals, dtype=torch.int64)

        timestamps, counts, used = parse_metadata_tensor(tensor)

        np.testing.assert_array_equal(
            timestamps.numpy(), np.array([0, 0xFFFFFFFF, 0], dtype=np.uint32)
        )
        np.testing.assert_array_equal(
            counts.numpy(), np.array([0, 0x7FFFFFFF, 0], dtype=np.uint32)
        )
        np.testing.assert_array_equal(
            used.numpy(), np.array([False, False, True], dtype=bool)
        )

    def test_invalid_dtype(self) -> None:
        """
        Test that an assertion is raised for wrong dtype.
        """
        tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
        with self.assertRaises(AssertionError):
            parse_metadata_tensor(tensor)


class GetKvZchEvictionMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        # Prepare some metadata values with timestamp, count, used
        # Use negative numbers to represent highest bit set (used=1)
        self.vals = [
            (100 | (5 << 32)),  # used=0
            (int(time.time()) - 60 | (10 << 32))
            - (1 << 63),  # used=1, timestamp 1 min ago
            (int(time.time()) - 3600 | (15 << 32))
            - (1 << 63),  # used=1, timestamp 1 hour ago
        ]
        self.metadata_tensor = torch.tensor(self.vals, dtype=torch.int64)

    def test_count_based_eviction(self) -> None:
        policy = CountBasedEvictionPolicy(inference_eviction_threshold=10)
        mask = get_kv_zch_eviction_mask(self.metadata_tensor, policy)
        # counts are 5,10,15; threshold=10; keep counts >= 10
        expected = torch.tensor([False, True, True], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected))

    def test_timestamp_based_eviction(self) -> None:
        policy = TimestampBasedEvictionPolicy(inference_eviction_ttl_mins=30)
        mask = get_kv_zch_eviction_mask(self.metadata_tensor, policy)
        # timestamps: 100 (old), now-60s, now-3600s
        # TTL=30min=1800s, keep timestamps within 1800s
        expected = torch.tensor([False, True, False], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected))

    def test_count_timestamp_mixed_eviction(self) -> None:
        policy = CountTimestampMixedEvictionPolicy(
            inference_eviction_threshold=10, inference_eviction_ttl_mins=30
        )
        mask = get_kv_zch_eviction_mask(self.metadata_tensor, policy)
        # count mask: counts >= 10 -> [False, True, True]
        # timestamp mask: within 1800s -> [False, True, False]
        # combined mask = count_mask & timestamp_mask
        expected = torch.tensor([False, True, False], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected))

    def test_no_eviction_policy(self) -> None:
        policy = NoEvictionPolicy()
        mask = get_kv_zch_eviction_mask(self.metadata_tensor, policy)
        # No eviction, mask all True
        expected = torch.ones_like(self.metadata_tensor, dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected))

    def test_unsupported_policy(self) -> None:
        class DummyPolicy(VirtualTableEvictionPolicy):
            pass

        policy = DummyPolicy()
        with self.assertRaises(ValueError):
            get_kv_zch_eviction_mask(self.metadata_tensor, policy)
