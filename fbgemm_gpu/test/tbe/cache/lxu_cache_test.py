#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import random
import unittest
from itertools import accumulate
from typing import Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_utils import to_device
from fbgemm_gpu.split_table_batched_embeddings_ops_training import DEFAULT_ASSOC
from hypothesis import given, settings, Verbosity
from torch import Tensor

from .. import common  # noqa E402
from ..common import MAX_EXAMPLES, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, optests


VERBOSITY: Verbosity = Verbosity.verbose


@optests.generate_opcheck_tests(fast=True)
class LXUCacheTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(
        associativity=st.sampled_from([1, DEFAULT_ASSOC]),
    )
    @settings(deadline=None)
    def test_lxu_cache_lookup(self, associativity: int) -> None:
        max_index: int = 8000
        # Use single cache set to avoid dealing with cache set hash algorithm.
        lxu_cache_state_gpu = (
            torch.arange(associativity, dtype=torch.int64).unsqueeze(0).cuda()
        )

        # Testing all miss.
        linear_cache_indices_0 = (
            torch.tensor([32, 33, 34, 35, 36, 100, 1000, 1725])
            if associativity <= 32
            else torch.tensor([64, 65, 66, 67, 68, 100, 1000, 1725])
        ).cuda()
        lxu_locations = torch.ops.fbgemm.lxu_cache_lookup(
            linear_cache_indices_0, lxu_cache_state_gpu, max_index
        )
        torch.testing.assert_close(
            lxu_locations,
            torch.full_like(lxu_locations, -1),
        )

        # Testing all hits.
        cache_indices_1 = torch.randint(0, associativity, (associativity,))
        linear_cache_indices_1 = cache_indices_1.cuda()
        lxu_locations = torch.ops.fbgemm.lxu_cache_lookup(
            linear_cache_indices_1, lxu_cache_state_gpu, max_index
        )
        torch.testing.assert_close(
            lxu_locations.cpu(),
            cache_indices_1.int(),
        )

        # Testing mixture.
        miss_cache_indices_0 = torch.randint(associativity, max_index // 2, (10,))
        hit_cache_indices_0 = torch.randint(0, associativity, (8,))
        miss_cache_indices_1 = torch.randint(max_index // 2, max_index, (16,))
        hit_cache_indices_1 = torch.randint(0, associativity, (8,))
        linear_cache_indices_2 = torch.cat(
            [
                miss_cache_indices_0,
                hit_cache_indices_0,
                miss_cache_indices_1,
                hit_cache_indices_1,
            ]
        ).cuda()
        lxu_locations = torch.ops.fbgemm.lxu_cache_lookup(
            linear_cache_indices_2, lxu_cache_state_gpu, max_index
        )

        expected_result = torch.cat(
            [
                torch.full_like(miss_cache_indices_0, -1),
                hit_cache_indices_0,
                torch.full_like(miss_cache_indices_1, -1),
                hit_cache_indices_1,
            ]
        ).int()
        torch.testing.assert_close(
            lxu_locations.cpu(),
            expected_result,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        cache_sets=st.integers(min_value=10, max_value=300),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_lxu_cache_locking_counter_decrement(
        self,
        cache_sets: int,
    ) -> None:
        warp_size = DEFAULT_ASSOC
        N = cache_sets * warp_size
        lxu_cache_locking_counter = torch.randint(
            low=1,
            high=3,
            size=[cache_sets, warp_size],
            device="cuda",
            dtype=torch.int32,
        )
        counter_ref = lxu_cache_locking_counter.tolist()
        lxu_cache_locations_list = []
        lxu_cache_locations_set = set()
        for _ in range(3 * N):
            location = random.randrange(-1, N)
            lxu_cache_locations_list.append(location)
            lxu_cache_locations_set.add(location)

        for idx in lxu_cache_locations_set:
            if idx >= 0:
                q, r = idx // warp_size, idx % warp_size
                counter_ref[q][r] -= 1

        counter_ref = torch.tensor(counter_ref, device="cuda", dtype=torch.int32)
        lxu_cache_locations = torch.tensor(
            lxu_cache_locations_list, device="cuda", dtype=torch.int32
        )
        torch.ops.fbgemm.lxu_cache_locking_counter_decrement(
            lxu_cache_locking_counter, lxu_cache_locations
        )
        self.assertTrue(torch.equal(lxu_cache_locking_counter, counter_ref))

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_unique_lxu_cache_lookup(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
    ) -> None:
        E = int(10**log_E)

        indices = to_device(
            torch.randint(low=0, high=E, size=(T * L * B,)),
            use_cpu=False,
        ).long()
        offsets = to_device(
            torch.tensor([0] + list(accumulate([L] * (T * L)))),
            use_cpu=False,
        ).long()

        def unique_lookup(
            indices: Tensor,
            offsets: Tensor,
            cache_hash_size_cumsum: Tensor,
            total_cache_hash_size: int,
        ) -> Tuple[Tensor, Tensor]:
            linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
                cache_hash_size_cumsum,
                indices,
                offsets,
            )

            uniq_indices, uniq_indices_length, _ = torch.ops.fbgemm.get_unique_indices(
                linear_cache_indices, total_cache_hash_size, compute_count=False
            )

            uniq_lxu_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
                uniq_indices,
                lxu_cache_state,
                total_cache_hash_size,
                gather_cache_stats=False,
                num_uniq_cache_indices=uniq_indices_length,
            )

            return uniq_lxu_cache_locations, uniq_indices_length

        def duplicate_lookup(
            indices: Tensor,
            offsets: Tensor,
            cache_hash_size_cumsum: Tensor,
            total_cache_hash_size: int,
        ) -> Tensor:
            linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
                cache_hash_size_cumsum,
                indices,
                offsets,
            )

            lxu_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
                linear_cache_indices,
                lxu_cache_state,
                total_cache_hash_size,
                gather_cache_stats=False,
            )
            return lxu_cache_locations

        cache_sets = int((E * T) * 0.2)
        lxu_cache_state = torch.zeros(
            cache_sets,
            DEFAULT_ASSOC,
            device="cuda",
            dtype=torch.int64,
        ).fill_(-1)

        hash_sizes = torch.tensor([E] * T, dtype=torch.long, device="cuda")
        cache_hash_size_cumsum = torch.ops.fbgemm.asynchronous_complete_cumsum(
            hash_sizes
        )
        total_cache_hash_size = cache_hash_size_cumsum[-1].item()

        linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum,
            indices,
            offsets,
        )

        # Emulate cache population
        uniq_indices_cpu = linear_cache_indices.unique().cpu()
        index_cache_set_map = uniq_indices_cpu.clone()
        index_cache_set_map.apply_(
            lambda x: torch.ops.fbgemm.lxu_cache_slot(x, cache_sets)
        )
        index_cache_set_map = index_cache_set_map.tolist()
        uniq_indices_cpu = uniq_indices_cpu.tolist()

        slots = {}
        for idx, c in zip(uniq_indices_cpu, index_cache_set_map):
            if c not in slots:
                slots[c] = 0
            slot = slots[c]
            if slot < DEFAULT_ASSOC:
                lxu_cache_state[c][slot] = idx
            slots[c] = slot + 1

        # Run unique lookup
        uniq_lookup_output, uniq_indices_length = unique_lookup(
            indices, offsets, cache_hash_size_cumsum, total_cache_hash_size
        )

        # Run duplicate lookup
        duplicate_lookup_output = duplicate_lookup(
            indices, offsets, cache_hash_size_cumsum, total_cache_hash_size
        )

        # Start running validation

        # Compute unique indices using PyTorch ops
        sorted_linear_cache_indices, inverse_sorted_cache_indices = torch.sort(
            linear_cache_indices
        )
        ref_uniq_cache_indices, cache_indices_counts = torch.unique_consecutive(
            sorted_linear_cache_indices, return_inverse=False, return_counts=True
        )

        # Convert to lists
        cache_indices_counts = cache_indices_counts.cpu().tolist()
        uniq_lookup_output = uniq_lookup_output.cpu().tolist()

        # Validate the number of unique cache indices
        ref_num_uniq_indices = ref_uniq_cache_indices.numel()
        assert ref_num_uniq_indices == uniq_indices_length.item()

        # Expand
        reshaped_uniq_lookup_output = uniq_lookup_output[:ref_num_uniq_indices]
        sorted_lxu_cache_locations = to_device(
            torch.tensor(
                np.repeat(reshaped_uniq_lookup_output, cache_indices_counts),
                dtype=duplicate_lookup_output.dtype,
            ),
            use_cpu=False,
        )

        _, cache_location_indices = torch.sort(inverse_sorted_cache_indices)

        expanded_lxu_cache_locations = torch.index_select(
            sorted_lxu_cache_locations, 0, cache_location_indices
        )

        assert torch.equal(expanded_lxu_cache_locations, duplicate_lookup_output)


if __name__ == "__main__":
    unittest.main()
