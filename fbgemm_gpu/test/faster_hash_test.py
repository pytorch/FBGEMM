#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from enum import IntEnum

# pyre-ignore[21]
import fbgemm_gpu  # noqa: F401

import torch

# check if we are in open source env to decide how to import necessary modules
try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import (  # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
        gpu_unavailable,
        skipIfRocm,
    )
except Exception:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, skipIfRocm

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:faster_hash_ops")


class HashZchKernelEvictionPolicy(IntEnum):
    THRESHOLD_EVICTION = 0
    LRU_EVICTION = 1


class FasterHashTest(unittest.TestCase):
    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_simple_zch_no_evict(self) -> None:
        """
        Test the basic functionality of zero collision hash without evicting.
        It creates a identity table with 200 slots, and insert two batches of
        100 numbers with zero collision hash.

        Assertions:
            1. The outputs of two batches match their inputs.
            2. The identity table is fully utilized.
            3. The readonly lookup matches the original output.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior.
        """
        # no evict
        identities, _ = torch.ops.fbgemm.create_zch_buffer(
            200, device=torch.device("cuda")
        )
        numbers = torch.arange(0, 100, dtype=torch.int64, device="cuda")
        local_sizes = torch.ones_like(numbers) * 100

        output1, evict_slots1 = torch.ops.fbgemm.zero_collision_hash(
            input=numbers,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            local_sizes=local_sizes,
            offsets=torch.zeros_like(numbers),
        )
        output2, evict_slots2 = torch.ops.fbgemm.zero_collision_hash(
            input=numbers + 100,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            local_sizes=local_sizes,
            offsets=torch.ones_like(numbers) * 100,
        )

        self.assertEqual(
            torch.unique(output1).tolist(),
            numbers.tolist(),
            f"{torch.unique(output1).tolist()=} != {numbers.tolist()=}",
        )
        self.assertEqual(torch.unique(output2).tolist(), (numbers + 100).tolist())
        self.assertTrue(torch.all(identities != -1))

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            input=numbers + 100,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
            local_sizes=local_sizes,
            offsets=torch.ones_like(numbers) * 100,
        )
        self.assertTrue(torch.equal(output2, output_readonly))

        # CPU
        output_readonly_cpu, _ = torch.ops.fbgemm.zero_collision_hash(
            input=numbers.cpu() + 100,
            identities=identities.cpu(),
            max_probe=100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
            local_sizes=local_sizes.cpu(),
            offsets=torch.ones_like(numbers).cpu() * 100,
        )
        self.assertTrue(
            torch.equal(output2.cpu(), output_readonly_cpu),
            f"{output2.cpu()=} != {output_readonly_cpu=}",
        )

        # other numbers.
        identities, _ = torch.ops.fbgemm.create_zch_buffer(
            100, device=torch.device("cuda")
        )
        numbers_100_200 = torch.arange(100, 200, dtype=torch.int64, device="cuda")
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_200,
            identities,
            100,
            circular_probe=True,
        )
        self.assertEqual(torch.unique(output).tolist(), numbers.tolist())
        self.assertTrue(torch.all(identities != -1))

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_200,
            identities,
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output, output_readonly))

        # CPU
        output_readonly_cpu, _ = torch.ops.fbgemm.zero_collision_hash(
            input=numbers_100_200.cpu(),
            identities=identities.cpu(),
            max_probe=100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output_readonly.cpu(), output_readonly_cpu))

        # no evict + no circular probe
        identities, _ = torch.ops.fbgemm.create_zch_buffer(
            100, device=torch.device("cuda")
        )
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers,
            identities,
            100,
            circular_probe=False,
        )
        self.assertFalse(torch.all(identities != -1))
        unique_indices = torch.unique(output)
        all_indices = torch.arange(identities.size(0), device="cuda")
        not_select_indices = torch.isin(all_indices, unique_indices, invert=True)
        self.assertTrue(torch.all(identities[unique_indices] != -1))
        self.assertTrue(torch.all(identities[not_select_indices] == -1))

        unique_elements, counts = torch.unique(
            identities[identities[:, 0] != -1][:, 0], return_counts=True
        )
        self.assertTrue(torch.all(counts == 1))

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers,
            identities,
            100,
            circular_probe=False,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output, output_readonly))

        # CPU
        output_readonly_cpu, _ = torch.ops.fbgemm.zero_collision_hash(
            input=numbers.cpu(),
            identities=identities.cpu(),
            max_probe=100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output_readonly.cpu(), output_readonly_cpu))

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_simple_zch_no_evict_rand(self) -> None:
        """
        Test the basic functionality of zero collision hash without evicting.
        It creates a identity table with 100 slots, and insert 100 random numbers
        with zero collision hash.

        Assertions:
            1. The outputs of two batches match their inputs.
            2. The identity table is fully utilized.
            3. The readonly lookup matches the original output.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior.
        """
        # no evict - rand number.
        identities, _ = torch.ops.fbgemm.create_zch_buffer(
            100, device=torch.device("cuda")
        )
        random_numbers = torch.randint(0, 100, (100,), device="cuda")
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            random_numbers,
            identities,
            100,
            circular_probe=True,
        )

        for i in range(100):
            to_test = output[random_numbers == i]
            if len(to_test) > 0:
                self.assertTrue(torch.all(to_test == to_test[0]))

        unique_indices = torch.unique(output)
        all_indices = torch.arange(identities.size(0), device="cuda")
        not_select_indices = torch.isin(all_indices, unique_indices, invert=True)
        self.assertTrue(torch.all(identities[unique_indices] != -1))
        self.assertTrue(torch.all(identities[not_select_indices] == -1))
        unique_elements, counts = torch.unique(
            identities[identities[:, 0] != -1][:, 0], return_counts=True
        )
        self.assertTrue(torch.all(counts == 1))

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            random_numbers,
            identities,
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output, output_readonly))

        # CPU
        output_readonly_cpu, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            random_numbers.cpu(),
            identities.cpu(),
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output.cpu(), output_readonly_cpu))

        # no evict + no circular probe
        identities, _ = torch.ops.fbgemm.create_zch_buffer(
            100, device=torch.device("cuda")
        )
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            random_numbers,
            identities,
            100,
            circular_probe=False,
        )
        unique_indices_no_circular = torch.unique(output)
        all_indices_no_circular = torch.arange(identities.size(0), device="cuda")
        not_select_indices_no_circular = torch.isin(
            all_indices_no_circular, unique_indices_no_circular, invert=True
        )
        self.assertTrue(torch.all(identities[unique_indices_no_circular] != -1))
        self.assertTrue(torch.all(identities[not_select_indices_no_circular] == -1))
        self.assertTrue(unique_indices_no_circular.size(0) <= unique_indices.size(0))
        unique_elements, counts = torch.unique(
            identities[identities[:, 0] != -1][:, 0], return_counts=True
        )
        self.assertTrue(torch.all(counts == 1))

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            random_numbers,
            identities,
            100,
            circular_probe=False,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output, output_readonly))

        # CPU
        output_readonly_cpu, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            random_numbers.cpu(),
            identities.cpu(),
            100,
            circular_probe=False,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output.cpu(), output_readonly_cpu))

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_simple_zch_evict(self) -> None:
        """
        Test the basic functionality of zero collision hash with evicting.
        It creates a identity table with 100 slots, and insert 100 numbers
        from 0 to 99 with zero collision hash. Then it inserts 100 numbers
        from 100 to 199 with zero collision hash. The last 100 numbers should
        evict the first 100 numbers.

        Assertions (besides the no-evict assertions):
            1. No evictions happen in the first batch.
            2. The evicted indices are the first 100 numbers.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior.
        """
        # evict
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            100, support_evict=True, device=torch.device("cuda")
        )
        numbers = torch.arange(0, 100, dtype=torch.int64, device="cuda")
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers,
            identities,
            100,
            circular_probe=True,
            exp_hours=7 * 24,
            metadata=metadata,
        )
        self.assertEqual(torch.unique(output).tolist(), numbers.tolist())
        self.assertTrue(evict_slots.numel() == 0)

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers,
            identities,
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output, output_readonly))

        # evict with all expired hours.
        metadata[:, 0] -= 7 * 24 + 1
        numbers_100_200 = torch.arange(100, 200, dtype=torch.int64, device="cuda")
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_200,
            identities,
            100,
            circular_probe=True,
            exp_hours=7 * 24,
            metadata=metadata,
        )
        self.assertEqual(torch.unique(output).tolist(), numbers.tolist())
        self.assertTrue(torch.all(evict_slots != -1))
        self.assertEqual(torch.unique(evict_slots).tolist(), numbers.tolist())

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_200,
            identities,
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output, output_readonly))

        # evict + no circular probe
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            100, support_evict=True, device=torch.device("cuda")
        )
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers,
            identities,
            100,
            circular_probe=False,
            exp_hours=7 * 24,
            metadata=metadata,
        )
        self.assertFalse(torch.all(identities != -1))
        unique_indices = torch.unique(output)
        all_indices = torch.arange(identities.size(0), device="cuda")
        not_select_indices = torch.isin(all_indices, unique_indices, invert=True)
        self.assertTrue(torch.all(identities[unique_indices] != -1))
        self.assertTrue(torch.all(identities[not_select_indices] == -1))
        unique_elements, counts = torch.unique(
            identities[identities[:, 0] != -1][:, 0], return_counts=True
        )
        self.assertTrue(torch.all(counts == 1))

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers,
            identities,
            100,
            circular_probe=False,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output, output_readonly))

        # evict with all expired hours + no circular probe
        evict_slot_candidate_mask = metadata[:, 0] != -1
        evict_slot_candidates = torch.nonzero(evict_slot_candidate_mask)
        self.assertTrue(evict_slot_candidates.size(0) != 0)
        metadata[evict_slot_candidate_mask, 0] -= 7 * 24 + 1
        old_time_value = metadata[evict_slot_candidates[0], 0]
        numbers_100_200 = torch.arange(100, 200, dtype=torch.int64, device="cuda")
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_200,
            identities,
            100,
            circular_probe=False,
            exp_hours=7 * 24,
            metadata=metadata,
        )
        self.assertTrue(torch.all(torch.isin(evict_slots, evict_slot_candidates)))
        self.assertTrue(torch.all(metadata[evict_slots][:, 0] != old_time_value))
        unique_elements, counts = torch.unique(
            identities[identities[:, 0] != -1][:, 0], return_counts=True
        )
        self.assertTrue(torch.all(counts == 1))

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_200,
            identities,
            100,
            circular_probe=False,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output, output_readonly))

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_simple_zch_evict_with_rand_unique_numbers(self) -> None:
        """
        Test the basic functionality of zero collision hash with evicting.
        It creates a identity table with 100 slots, and insert 100 random numbers
        with zero collision hash. Then it inserts 100 random numbers with zero
        collision hash. The last 100 numbers should evict the first 100 numbers.

        Assertions (besides the no-evict assertions):
            1. No evictions happen in the first batch.
            2. The evicted indices are the first 100 numbers.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior.
        """
        # evict - rand number.
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            100, support_evict=True, device=torch.device("cuda")
        )
        random_numbers = torch.unique(torch.randint(0, 100, (100,), device="cuda"))
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            random_numbers,
            identities,
            100,
            circular_probe=True,
            exp_hours=7 * 24,
            metadata=metadata,
        )

        for i in range(100):
            to_test = output[random_numbers == i]
            if len(to_test) > 0:
                self.assertTrue(torch.all(to_test == to_test[0]))

        unique_indices = torch.unique(output)
        all_indices = torch.arange(identities.size(0), device="cuda")
        not_select_indices = torch.isin(all_indices, unique_indices, invert=True)
        self.assertTrue(torch.all(identities[unique_indices] != -1))
        self.assertTrue(torch.all(identities[not_select_indices] == -1))
        unique_elements, counts = torch.unique(
            identities[identities[:, 0] != -1][:, 0], return_counts=True
        )
        self.assertTrue(torch.all(counts == 1))

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            random_numbers,
            identities[:, 0].unsqueeze(1),
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output, output_readonly))

        # evict - rand number + no circular probe
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            100, support_evict=True, device=torch.device("cuda")
        )
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            random_numbers,
            identities,
            100,
            circular_probe=False,
            exp_hours=7 * 24,
            metadata=metadata,
        )

        for i in range(100):
            to_test = output[random_numbers == i]
            if len(to_test) > 0:
                self.assertTrue(torch.all(to_test == to_test[0]))

        unique_indices_no_circular = torch.unique(output)
        all_indices_no_circular = torch.arange(identities.size(0), device="cuda")
        not_select_indices_no_circular = torch.isin(
            all_indices_no_circular, unique_indices_no_circular, invert=True
        )
        self.assertTrue(torch.all(identities[unique_indices_no_circular] != -1))
        self.assertTrue(torch.all(identities[not_select_indices_no_circular] == -1))
        self.assertTrue(unique_indices_no_circular.size(0) <= unique_indices.size(0))
        unique_elements, counts = torch.unique(
            identities[identities[:, 0] != -1][:, 0], return_counts=True
        )
        self.assertTrue(torch.all(counts == 1))

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            random_numbers,
            identities,
            100,
            circular_probe=False,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output, output_readonly))

        # evict with all expired hours + no circular probe
        evict_slot_candidate_mask = metadata[:, 0] != -1
        evict_slot_candidates = torch.nonzero(evict_slot_candidate_mask)
        self.assertTrue(evict_slot_candidates.size(0) != 0)
        metadata[evict_slot_candidate_mask, 0] -= 7 * 24 + 1
        old_time_value = metadata[evict_slot_candidates[0], 0]
        random_numbers_100_200 = torch.unique(
            torch.randint(100, 200, (100,), device="cuda")
        )
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            random_numbers_100_200,
            identities,
            100,
            circular_probe=False,
            exp_hours=7 * 24,
            metadata=metadata,
        )
        self.assertTrue(torch.all(torch.isin(evict_slots, evict_slot_candidates)))
        self.assertTrue(torch.all(metadata[evict_slots][:, 0] != old_time_value))

        unique_elements, counts = torch.unique(
            identities[identities[:, 0] != -1][:, 0], return_counts=True
        )
        self.assertTrue(torch.all(counts == 1), counts)

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            random_numbers_100_200,
            identities[:, 0].unsqueeze(1),
            100,
            circular_probe=False,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(
            torch.equal(output, output_readonly), f"{output=}, {output_readonly=}"
        )

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_eviction_during_lookup(self) -> None:
        """
        Test the basic functionality of zero collision hash with evicting.
        It creates a identity table with 100 slots, and insert 99 numbers
        with zero collision hash. Then it inserts 1 random number 101 with zero
        collision hash, then all the slots should be filled. Then it makes all
        the slots expired except for the 101 one, and insert the number 101 again
        with zero collision hash, then none slot should be evicted but all the identity
        table should only have one value. Then it inserts 1 random number 102 with zero
        collision hash, then one slot should be evicted.

        Assertions:
            1. After inserting 99 numbers, there is only one empty slot.
            2. After inserting 101, all the slots are filled.
            3. After making all the slots expired, and inserting 101 again, there is only one
                slot with value 101.
            4. After inserting 102, one slot should be evicted.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior.
        """
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            100, support_evict=True, device=torch.device("cuda")
        )
        numbers_0_99 = torch.arange(0, 99, dtype=torch.int64, device="cuda")
        output_0_99, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_99,
            identities,
            100,
            circular_probe=True,
            exp_hours=7 * 24,
            metadata=metadata,
        )
        empty_slots = identities[:, 0] == -1
        self.assertTrue(torch.sum(empty_slots) == 1, torch.sum(empty_slots))

        # insert number 101, should be able to fill all slots.
        numbers = torch.tensor([101], dtype=torch.int64, device="cuda")
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers,
            identities,
            100,
            circular_probe=True,
            exp_hours=7 * 24,
            metadata=metadata,
        )
        self.assertTrue(torch.all(identities[:, 0] != -1))

        # make none 101 slots expired.
        metadata[~empty_slots, 0] -= 7 * 24 + 1
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers,
            identities,
            100,
            circular_probe=True,
            exp_hours=7 * 24,
            metadata=metadata,
        )
        unique_elements, counts = torch.unique(identities[:, 0], return_counts=True)
        self.assertTrue(torch.all(counts == 1))
        self.assertTrue(evict_slots.numel() == 0)

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_99,
            identities,
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(torch.equal(output_0_99, output_readonly))

        # evict some slot.
        numbers = torch.tensor([102], dtype=torch.int64, device="cuda")
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers,
            identities,
            100,
            circular_probe=True,
            exp_hours=7 * 24,
            metadata=metadata,
        )
        self.assertTrue(evict_slots.numel() == 1)

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_zch_output_on_uvm(self) -> None:
        """
        Test the zero collision hash output on uvm.
        It creates a identity table with 200 slots, and insert 100 numbers
        with zero collision hash. Then it inserts 100 random numbers with zero
        collision hash. The output should be on uvm (cpu).

        Assertions:
            1. The output is on cpu.
            2. The output is correct.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior.
        """
        # no evict
        identities, _ = torch.ops.fbgemm.create_zch_buffer(
            200, device=torch.device("cuda")
        )
        numbers = torch.arange(0, 100, dtype=torch.int64, device="cuda")

        output, _ = torch.ops.fbgemm.zero_collision_hash(
            input=numbers,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            output_on_uvm=True,
        )

        self.assertTrue(output.device.type == "cpu")

        add_on = torch.arange(100, 200, dtype=torch.int64)
        self.assertTrue(
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `int`.
            torch.equal(((output + add_on) + (output - add_on)), 2 * output)
        )

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_zch_int64_nohash_identity(self) -> None:
        """
        Test with the capability of procesing int64_t data for
        zero collision hash.

        Assertions:
            1. The output is correct for zero collision hash.
            2. The eviction works correctly when processing int64_t data.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior.
        """
        # no evict
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            100, device=torch.device("cuda"), support_evict=True, long_type=True
        )
        numbers = torch.arange(2**33, 2**33 + 100, dtype=torch.int64, device="cuda")

        output, _ = torch.ops.fbgemm.zero_collision_hash(
            input=numbers,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            readonly=False,
            exp_hours=7 * 24,
            metadata=metadata,
        )

        self.assertTrue(
            torch.equal(
                torch.sort(identities[identities != -1].view(-1))[0],
                numbers,
            ),
            f"{identities=} vs {numbers=}",
        )

        numbers_100_200 = torch.arange(
            2**33 + 100, 2**33 + 200, dtype=torch.int64, device="cuda"
        )
        metadata[:, 0] -= 7 * 24 + 1
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            input=numbers_100_200,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            readonly=False,
            exp_hours=7 * 24,
            metadata=metadata,
        )

        expect_indices = list(range(100))
        self.assertEqual(torch.unique(output).tolist(), expect_indices)
        self.assertTrue(torch.all(evict_slots != -1))
        self.assertEqual(torch.unique(evict_slots).tolist(), expect_indices)
        self.assertTrue(
            torch.equal(
                torch.sort(identities[identities != -1].view(-1))[0],
                numbers_100_200,
            ),
            f"{identities=} vs {numbers_100_200=}",
        )

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_zch_int32_nohash_identity(self) -> None:
        """
        Test with the capability of procesing int32_t data for
        zero collision hash.

        Assertions:
            1. The output is correct for zero collision hash.
            2. The eviction works correctly when processing int32_t data.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior.
        """
        # no evict
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            100, device=torch.device("cuda"), support_evict=True, long_type=False
        )
        numbers = torch.arange(2**33, 2**33 + 100, dtype=torch.int32, device="cuda")

        output, _ = torch.ops.fbgemm.zero_collision_hash(
            input=numbers,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            readonly=False,
            exp_hours=7 * 24,
            metadata=metadata,
        )

        self.assertTrue(
            torch.equal(
                torch.sort(identities[identities != -1].view(-1))[0],
                numbers,
            ),
            f"{identities=} vs {numbers=}",
        )

        numbers_100_200 = torch.arange(
            2**33 + 100, 2**33 + 200, dtype=torch.int32, device="cuda"
        )
        metadata[:, 0] -= 7 * 24 + 1
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            input=numbers_100_200,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            readonly=False,
            exp_hours=7 * 24,
            metadata=metadata,
        )

        expect_indices = list(range(100))
        self.assertEqual(torch.unique(output).tolist(), expect_indices)
        self.assertTrue(torch.all(evict_slots != -1))
        self.assertEqual(torch.unique(evict_slots).tolist(), expect_indices)
        self.assertTrue(
            torch.equal(
                torch.sort(identities[identities != -1].view(-1))[0],
                numbers_100_200,
            ),
            f"{identities=} vs {numbers_100_200=}",
        )

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_fallback(self) -> None:
        """
        Test the fallback functionality of zero collision hash.
        It creates a identity table with 100 slots, and insert 100 numbers
        with zero collision hash. Then it inserts 20 random numbers 90-109 with zero
        collision hash when all the slots should be filled. When enabling fallback,
        all the ids (including unexisting ones) are mapped to a position. When disabling
        fallback, existing ids are mapped to a position and unexisting ones are mapped to -1.

        Assertions:
            1. When fallback is enabled, all the ids (including unexisting ones) are mapped to a position.
            2. When fallback is disabled, existing ids are mapped to a position and unexisting ones are mapped to -1.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior.
        """
        # init and add some ids
        identities, _ = torch.ops.fbgemm.create_zch_buffer(
            100, device=torch.device("cuda"), long_type=True
        )
        ids = torch.arange(0, 100, device="cuda")
        output, _ = torch.ops.fbgemm.zero_collision_hash(
            input=ids,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            readonly=False,
        )

        # non-readonly and fallback enabled
        ids = torch.arange(90, 120, device="cuda")
        remapped_ids, _ = torch.ops.fbgemm.zero_collision_hash(
            input=ids,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            readonly=False,
            disable_fallback=False,
        )
        # all ids (including unexisting ones) are mapped to a position
        self.assertTrue(torch.all(remapped_ids != -1))

        # readonly and fallback enabled
        ids = torch.arange(90, 120, device="cuda")
        remapped_ids, _ = torch.ops.fbgemm.zero_collision_hash(
            input=ids,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            readonly=True,
            disable_fallback=False,
        )
        # all ids (including unexisting ones) are mapped to a position
        self.assertTrue(torch.all(remapped_ids != -1))

        # non-readonly and fallback disabled
        ids = torch.arange(90, 120, device="cuda")
        remapped_ids, _ = torch.ops.fbgemm.zero_collision_hash(
            input=ids,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            readonly=False,
            disable_fallback=True,
        )
        # existing ids are mapped to a position and unexisting ones are mapped to -1
        self.assertTrue(
            torch.equal(
                torch.index_select(
                    identities, 0, remapped_ids[remapped_ids != -1]
                ).squeeze(),
                torch.arange(90, 100, device="cuda"),
            )
        )
        self.assertTrue(torch.all(remapped_ids[-20:] == -1))

        # readonly and fallback disabled
        ids = torch.arange(90, 120, device="cuda")
        remapped_ids, _ = torch.ops.fbgemm.zero_collision_hash(
            input=ids,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            readonly=True,
            disable_fallback=True,
        )
        # existing ids are mapped to a position and unexisting ones are mapped to -1
        self.assertTrue(
            torch.equal(
                torch.index_select(
                    identities, 0, remapped_ids[remapped_ids != -1]
                ).squeeze(),
                torch.arange(90, 100, device="cuda"),
            )
        )
        self.assertTrue(torch.all(remapped_ids[-20:] == -1))

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_simple_zch_individual_score_evict(self) -> None:
        """
        Test the zero collision hash with individual score evict.
        It creates a identity table with 100 slots, and insert 100 numbers
        with zero collision hash. Then it sets a threshold to make half of the
        slots evictable. Then it inserts 100 random numbers with zero collision
        hash, and check the number of evicted slots. Then it looks up with the
        values between 0 and 99, and check the output is correct.

        Assertions:
            1. After inserting 100 numbers, there is none empty slot, and the remmaped ids are in range [0, 99].
            2. The metadata for the inserted ids are mapped correctly.
            3. None eviction happens after inserting 100 numbers.
            4. Readonly lookup works correctly.
            5. After setting the threshold to make half of the slots evictable, and inserting 100 random numbers,
                half (50) of the slots are evicted.
            6. The metadata for the evicted ids are set correctly.
            7. The metadata for the inserted ids are mapped correctly.
            8. The read-only output values for the requried 0-99 are correct.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior.
        """
        # evict
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            100, support_evict=True, long_type=True, device=torch.device("cuda")
        )
        numbers_0_100 = torch.arange(0, 100, dtype=torch.int64, device="cuda")
        input_metadata_500_600 = torch.arange(
            500, 600, dtype=torch.int32, device="cuda"
        )
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_100,
            identities,
            100,
            circular_probe=True,
            metadata=metadata,
            input_metadata=input_metadata_500_600,
            eviction_threshold=100,
        )
        self.assertEqual(torch.unique(output).tolist(), numbers_0_100.tolist())
        self.assertEqual(
            torch.unique(metadata).tolist(), input_metadata_500_600.tolist()
        )
        self.assertTrue(evict_slots.numel() == 0)

        # readonly lookup.
        output_readonly, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_100,
            identities,
            100,
            circular_probe=True,
            readonly=True,
        )
        self.assertTrue(torch.equal(output, output_readonly))

        numbers_100_200 = torch.arange(100, 200, dtype=torch.int64, device="cuda")
        input_metadata_600_700 = torch.arange(
            600, 700, dtype=torch.int32, device="cuda"
        )

        # evict by setting eviction_threshold to 550 (half of the slots of which the
        # eviction scores are less 550 will be evicted)
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_200,
            identities,
            100,
            circular_probe=True,
            metadata=metadata,
            input_metadata=input_metadata_600_700,
            eviction_threshold=550,
        )

        self.assertEqual(evict_slots.numel(), 50)
        self.assertTrue(torch.all(metadata >= 550))

        # readonly lookup.
        output_readonly, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_200,
            identities,
            100,
            circular_probe=True,
            readonly=True,
        )
        self.assertTrue(torch.equal(output, output_readonly))

        # attempt to update with lower input_metadata values
        metadata0 = metadata.clone()
        input_metadata_0_100 = torch.arange(0, 100, dtype=torch.int32, device="cuda")
        output_lower_metadata, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_200,
            identities,
            100,
            circular_probe=True,
            metadata=metadata,
            input_metadata=input_metadata_0_100,
            eviction_threshold=550,
        )

        self.assertTrue(torch.equal(output_lower_metadata, output))
        # metadata should not be overwritten
        self.assertTrue(torch.equal(metadata, metadata0))

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_zch_lru_evict(self) -> None:
        """
        Test the zero collision hash with LRU eviction.

        Tested eviction policy: HashZchKernelEvictionPolicy.LRU_EVICTION.value
        """
        # No evict
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            100, support_evict=True, device=torch.device("cuda")
        )
        numbers_0_100 = torch.arange(0, 100, dtype=torch.int64, device="cuda")

        cur_hour = 500
        ttl = 72

        input_metadata = torch.full_like(
            numbers_0_100,
            ttl + cur_hour,
            dtype=torch.int32,
            device="cuda",
        )

        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_100,
            identities,
            100,
            circular_probe=True,
            metadata=metadata,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
            input_metadata=input_metadata,
            eviction_threshold=cur_hour,
        )
        self.assertEqual(
            torch.unique(output).tolist(), numbers_0_100.tolist(), f"{output=}"
        )
        self.assertTrue(torch.all(metadata != -1), metadata)
        self.assertTrue(evict_slots.numel() == 0)
        self.assertEqual(
            torch.unique(identities).tolist(), numbers_0_100.tolist(), f"{identities=}"
        )

        # readonly lookup.
        output_readonly, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_100,
            identities,
            100,
            circular_probe=True,
            readonly=True,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
        )
        self.assertTrue(output.tolist(), output_readonly.tolist())

        output_readonly_cpu, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_100.cpu(),
            identities.cpu(),
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
        )
        self.assertTrue(
            torch.equal(output_readonly_cpu, output_readonly.cpu()),
            f"{output_readonly_cpu=} v.s {output_readonly.cpu()=}",
        )

        numbers_100_120 = torch.arange(100, 120, dtype=torch.int64, device="cuda")
        new_cur_hour = 600
        new_input_metadata = torch.full_like(
            numbers_100_120,
            ttl + new_cur_hour,
            dtype=torch.int32,
            device="cuda",
        )

        # modify metadata to set different update hours to trigger LRU eviction
        metadata = torch.randint(
            500, (100, 1), dtype=torch.int32, device=metadata.device
        )

        # arrange metadata in update order
        eviction_order = (
            torch.sort(metadata, 0)
            .indices.index_select(1, torch.tensor([0], device=metadata.device))
            .squeeze()
        )

        # all rows were occupied, do evict for all input numbers
        # evict by LRU
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            input=numbers_100_120,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            metadata=metadata,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
            input_metadata=new_input_metadata,
            eviction_threshold=new_cur_hour,
        )
        self.assertEqual(evict_slots.numel(), 20)
        self.assertTrue(
            set(evict_slots.tolist()).issubset(set(eviction_order[:40].tolist())),
            f"{evict_slots=}, {eviction_order=}",
        )

        self.assertTrue(
            torch.equal(
                torch.sort(identities[identities >= 100])[0],
                torch.sort(numbers_100_120)[0],
            ),
            f"{identities=} vs {numbers_100_120=}",
        )

        self.assertTrue(
            torch.equal(evict_slots, torch.sort(output)[0]),
            f"{evict_slots=} vs {output=}",
        )
        self.assertTrue(
            torch.equal(
                torch.nonzero(metadata >= 500), torch.nonzero(identities >= 100)
            ),
            f"{torch.nonzero(metadata >= 500)=} vs {torch.nonzero(identities >= 100)=}",
        )

        # readonly lookup again
        output_readonly, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_120,
            identities,
            100,
            circular_probe=True,
            readonly=True,
        )
        self.assertTrue(output.tolist(), output_readonly.tolist())

        output_readonly_cpu, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_120.cpu(),
            identities.cpu(),
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(
            torch.equal(output_readonly_cpu, output_readonly.cpu()),
            f"{output_readonly_cpu=} v.s {output_readonly.cpu()=}",
        )

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_zch_lru_evict_with_unexpired_slots(self) -> None:
        """
        Test the zero collision hash with LRU eviction with unexpired slots.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior,
                an assertion error will be raised.
        """
        # No evict
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            100, support_evict=True, device=torch.device("cuda")
        )
        numbers_0_100 = torch.arange(0, 100, dtype=torch.int64, device="cuda")

        cur_hour = 1000
        ttl = 72

        input_metadata = torch.full_like(
            numbers_0_100,
            ttl + cur_hour,
            dtype=torch.int32,
            device="cuda",
        )

        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_100,
            identities,
            100,
            circular_probe=True,
            metadata=metadata,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
            eviction_threshold=cur_hour,
            input_metadata=input_metadata,
        )
        self.assertEqual(
            torch.unique(output).tolist(), numbers_0_100.tolist(), f"{output=}"
        )
        self.assertTrue(torch.all(metadata != -1), metadata)
        self.assertTrue(evict_slots.numel() == 0)
        self.assertEqual(
            torch.unique(identities).tolist(), numbers_0_100.tolist(), f"{identities=}"
        )

        # readonly lookup.
        output_readonly, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_100,
            identities,
            100,
            circular_probe=True,
            readonly=True,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
        )
        self.assertTrue(output.tolist(), output_readonly.tolist())

        output_readonly_cpu, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_100.cpu(),
            identities.cpu(),
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
        )
        self.assertTrue(
            torch.equal(output_readonly_cpu, output_readonly.cpu()),
            f"{output_readonly_cpu=} v.s {output_readonly.cpu()=}",
        )

        numbers_100_150 = torch.arange(100, 150, dtype=torch.int64, device="cuda")

        # 20 slots expired, 80 unexpired
        metadata_to_update = torch.randint(
            500, 1050, (20, 1), dtype=torch.int32, device=metadata.device
        )
        metadata[0:20] = metadata_to_update

        metadata_index_0_20 = torch.arange(
            0, 20, dtype=torch.int64, device=metadata.device
        )

        new_cur_hour = 1050
        new_input_metadata = torch.full_like(
            numbers_100_150,
            ttl + new_cur_hour,
            dtype=torch.int32,
            device="cuda",
        )

        # all rows were occupied, do evict by LRU + TTL rule
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            input=numbers_100_150,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            metadata=metadata,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
            eviction_threshold=new_cur_hour,
            input_metadata=new_input_metadata,
        )
        self.assertEqual(evict_slots.numel(), 20)
        self.assertTrue(
            torch.equal(
                torch.sort(evict_slots)[0],
                torch.sort(metadata_index_0_20)[0],
            ),
            f"{evict_slots=}, {metadata_index_0_20=}",
        )

        self.assertTrue(torch.all(metadata[0:20][0] == 1050 + ttl))
        self.assertTrue(torch.all(metadata[20:][0] == 1000 + ttl))

        self.assertEqual(identities[identities >= 100].numel(), 20)
        self.assertTrue(torch.all(identities[20:][0] < 100))

        # readonly lookup - gpu
        output_readonly, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_150,
            identities,
            100,
            circular_probe=True,
            readonly=True,
        )
        self.assertTrue(output.tolist(), output_readonly.tolist())

        # readonly lookup - cpu
        output_readonly_cpu, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_100_150.cpu(),
            identities.cpu(),
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(
            torch.equal(output_readonly_cpu, output_readonly.cpu()),
            f"{output_readonly_cpu=} v.s {output_readonly.cpu()=}",
        )

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_rand_numbers_zch_lru_evict(self) -> None:
        """
        Test the zero collision hash with LRU eviction with random input numbers.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior,
                an assertion error will be raised.
        """
        # No evict
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            100, support_evict=True, device=torch.device("cuda"), long_type=True
        )
        numbers_0_100 = torch.arange(0, 100, dtype=torch.int64, device="cuda")

        cur_hour = 1000
        ttl = 24

        input_metadata = torch.full_like(
            numbers_0_100,
            ttl + cur_hour,  # TTL 24h
            dtype=torch.int32,
            device="cuda",
        )

        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_100,
            identities,
            100,
            circular_probe=True,
            metadata=metadata,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
            eviction_threshold=cur_hour,
            input_metadata=input_metadata,
        )

        self.assertEqual(torch.unique(output).tolist(), numbers_0_100.tolist())
        self.assertTrue(evict_slots.numel() == 0)

        # a tensor with 60 numbers with duplicates
        random_numbers_100_150 = torch.randint(
            100, 150, (60,), dtype=torch.int64, device="cuda"
        )

        new_cur_hour = 1025
        new_input_metadata = torch.full_like(
            random_numbers_100_150,
            ttl + new_cur_hour,
            dtype=torch.int32,
            device="cuda",
        )

        # all rows were occupied, do evict for all input numbers
        # evict by LRU
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            input=random_numbers_100_150,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            metadata=metadata,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
            eviction_threshold=new_cur_hour,
            input_metadata=new_input_metadata,
        )

        self.assertLessEqual(evict_slots.numel(), 60)
        self.assertTrue(
            torch.equal(
                torch.unique(identities[identities >= 100]),
                torch.unique(random_numbers_100_150),
            ),
            f"{torch.unique(identities[identities >= 100])=} vs {torch.unique(random_numbers_100_150)=}",
        )

        self.assertTrue(
            torch.equal(
                torch.nonzero(metadata >= 1025), torch.nonzero(identities >= 100)
            ),
            f"{torch.nonzero(metadata >= 1025)=} vs {torch.nonzero(identities >= 100)=}",
        )

        # readonly lookup again
        output_readonly, _ = torch.ops.fbgemm.zero_collision_hash(
            random_numbers_100_150,
            identities,
            100,
            circular_probe=True,
            readonly=True,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
        )
        self.assertTrue(output.tolist(), output_readonly.tolist())

        output_readonly_cpu, _ = torch.ops.fbgemm.zero_collision_hash(
            random_numbers_100_150.cpu(),
            identities.cpu(),
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
        )
        self.assertTrue(
            torch.equal(output_readonly_cpu, output_readonly.cpu()),
            f"{output_readonly_cpu=} v.s {output_readonly.cpu()=}",
        )

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_zch_lru_evict_with_offsets(self) -> None:
        """
        Test the zero collision hash with LRU eviction with offsets.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior,
                an assertion error will be raised.

        Skips:
            The test will be skipped if the GPU is not available, or not running on a CUDA device.
        """
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            200,
            device=torch.device("cuda"),
            long_type=True,
            support_evict=True,
        )

        numbers_0_100 = torch.arange(0, 100, dtype=torch.int64, device="cuda")
        local_sizes = torch.ones_like(numbers_0_100) * 100

        cur_hour = 1000
        ttl = 24
        input_metadata = torch.full_like(
            numbers_0_100,
            ttl + cur_hour,  # TTL 24h
            dtype=torch.int32,
            device="cuda",
        )

        output1, evict_slots1 = torch.ops.fbgemm.zero_collision_hash(
            input=numbers_0_100,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            metadata=metadata,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
            input_metadata=input_metadata,
            eviction_threshold=cur_hour,
            local_sizes=local_sizes,
            offsets=torch.zeros_like(numbers_0_100),
        )

        output2, evict_slots2 = torch.ops.fbgemm.zero_collision_hash(
            input=numbers_0_100 + 100,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            metadata=metadata,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
            input_metadata=input_metadata,
            eviction_threshold=cur_hour,
            local_sizes=local_sizes,
            offsets=torch.ones_like(numbers_0_100) * 100,
        )

        self.assertEqual(
            torch.unique(output1).tolist(),
            numbers_0_100.tolist(),
            f"{torch.unique(output1).tolist()=} != {numbers_0_100.tolist()=}",
        )

        self.assertEqual(torch.unique(output2).tolist(), (numbers_0_100 + 100).tolist())
        # verify all the rows in each batch are occupied
        self.assertTrue(torch.all(identities[0:99][0] != -1))
        self.assertTrue(torch.all(identities[100:199][0] != -1))

        # no eviction
        self.assertTrue(evict_slots1.numel() == 0)
        self.assertTrue(evict_slots2.numel() == 0)

        # readonly lookup.
        output_readonly, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            input=numbers_0_100 + 100,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
            local_sizes=local_sizes,
            offsets=torch.ones_like(numbers_0_100) * 100,
        )
        self.assertTrue(torch.equal(output2, output_readonly))

        # a tensor with 60 numbers with duplicates
        random_numbers_200_250 = torch.randint(
            200, 250, (60,), dtype=torch.int64, device="cuda"
        )
        # second input batch
        random_numbers_300_350 = random_numbers_200_250 + 100

        # modify metadata to set different timestamps in the range of [500, 1024)
        metadata = torch.randint(
            500, 1024, (200, 1), dtype=torch.int32, device=metadata.device
        )
        new_cur_hour = 1025
        new_input_metadata = torch.full_like(
            random_numbers_200_250,
            ttl + new_cur_hour,  # TTL 24h
            dtype=torch.int32,
            device="cuda",
        )

        local_sizes2 = torch.ones_like(random_numbers_200_250) * 100
        # all rows were occupied, do evict for all input numbers
        # evict by LRU
        output3, evict_slots3 = torch.ops.fbgemm.zero_collision_hash(
            input=random_numbers_200_250,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            metadata=metadata,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
            input_metadata=new_input_metadata,
            eviction_threshold=new_cur_hour,
            local_sizes=local_sizes2,
            offsets=torch.zeros_like(random_numbers_200_250),
        )

        output4, evict_slots4 = torch.ops.fbgemm.zero_collision_hash(
            input=random_numbers_300_350,
            identities=identities,
            max_probe=100,
            circular_probe=True,
            metadata=metadata,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
            input_metadata=new_input_metadata,
            eviction_threshold=new_cur_hour,
            local_sizes=local_sizes2,
            offsets=torch.ones_like(random_numbers_300_350) * 100,
        )

        # num of evicted slots may be < 60 when all slots were being probed/locked by other ids, this id will fall back to original slot (collide) and no eviction
        self.assertLessEqual(evict_slots3.numel(), 60)
        self.assertLessEqual(evict_slots4.numel(), 60)

        # verify index stored in evict_slot/output should within each batch's boundary
        self.assertTrue(torch.all(evict_slots3 < 100) and torch.all(evict_slots3 >= 0))
        self.assertTrue(
            torch.all(evict_slots4 < 200) and torch.all(evict_slots4 >= 100)
        )
        self.assertTrue(torch.all(output3 < 100) and torch.all(output3 >= 0))
        self.assertTrue(torch.all(output4 < 200) and torch.all(output4 >= 100))

        self.assertTrue(
            set(evict_slots3.tolist()).issubset(set(output3.tolist())),
            f"{evict_slots3=}, {torch.sort(output3)[0]=}",
        )
        self.assertTrue(
            set(evict_slots4.tolist()).issubset(set(output4.tolist())),
            f"{evict_slots4=}, {torch.sort(output4)[0]=}",
        )

        # verify values stored in identities
        first_half = identities.view(-1)[0:99]
        second_half = identities.view(-1)[100:199]
        self.assertTrue(
            set(first_half[first_half >= 200].tolist()).issubset(
                set(random_numbers_200_250.tolist())
            ),
            f"{set(first_half[first_half >= 200].tolist())=}, {set(random_numbers_200_250.tolist())=}",
        )
        self.assertTrue(
            set(second_half[second_half >= 200].tolist()).issubset(
                set(random_numbers_300_350.tolist())
            ),
            f"{set(second_half[second_half >= 300].tolist())=}, {set(random_numbers_300_350.tolist())=}",
        )

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_opt_in_with_prob(self) -> None:
        """
        Test the zero collision hash with opt-in with probability.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior,
                an assertion error will be raised.
        """
        zch_size = 100
        num_reserved_slots = 10
        num_opt_in_slots = zch_size - num_reserved_slots
        opt_in_prob = 20

        # without eviction
        identities, _ = torch.ops.fbgemm.create_zch_buffer(
            zch_size, support_evict=False, long_type=True, device=torch.device("cuda")
        )
        numbers = torch.arange(0, 100, dtype=torch.int64, device="cuda")
        opt_in_rands = torch.arange(0, 100, dtype=torch.int32, device="cuda")
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers,
            identities,
            100,
            circular_probe=True,
            opt_in_prob=opt_in_prob,
            num_reserved_slots=num_reserved_slots,
            opt_in_rands=opt_in_rands,
        )

        self.assertTrue(torch.sum((output >= 0) & (output < num_opt_in_slots)) == 20)
        self.assertTrue(
            torch.sum((output >= num_opt_in_slots) & (output < zch_size)) == 80
        )
        identities_opt_in_slots = identities[:num_opt_in_slots]
        identities_opt_in_slots_occupied = identities_opt_in_slots[
            identities_opt_in_slots != -1
        ]
        self.assertTrue(
            torch.equal(
                torch.unique(identities_opt_in_slots_occupied),
                torch.arange(0, 20, dtype=torch.int64, device="cuda"),
            )
        )
        identities_reserved_slots = identities[num_opt_in_slots:]
        self.assertTrue(torch.all(identities_reserved_slots == -1))

        # with eviction
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            zch_size, support_evict=True, long_type=True, device=torch.device("cuda")
        )
        numbers = torch.arange(0, 100, dtype=torch.int64, device="cuda")
        opt_in_rands = torch.arange(0, 100, dtype=torch.int32, device="cuda")
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers,
            identities,
            100,
            circular_probe=True,
            exp_hours=7 * 24,
            metadata=metadata,
            opt_in_prob=opt_in_prob,
            num_reserved_slots=num_reserved_slots,
            opt_in_rands=opt_in_rands,
        )

        self.assertTrue(torch.sum((output >= 0) & (output < num_opt_in_slots)) == 20)
        self.assertTrue(
            torch.sum((output >= num_opt_in_slots) & (output < zch_size)) == 80
        )
        identities_opt_in_slots = identities[:num_opt_in_slots]
        identities_opt_in_slots_occupied = identities_opt_in_slots[
            identities_opt_in_slots != -1
        ]
        self.assertTrue(
            torch.equal(
                torch.unique(identities_opt_in_slots_occupied),
                torch.arange(0, 20, dtype=torch.int64, device="cuda"),
            )
        )
        identities_reserved_slots = identities[num_opt_in_slots:]
        self.assertTrue(torch.all(identities_reserved_slots == -1))

        # readonly lookup
        numbers_0_20 = torch.arange(0, 20, dtype=torch.int64, device="cuda")
        output_readonly, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_20,
            identities,
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
            opt_in_prob=opt_in_prob,
            num_reserved_slots=num_reserved_slots,
        )
        self.assertTrue(
            torch.all((output_readonly >= 0) & (output_readonly < num_opt_in_slots))
        )
        output_readonly_cpu, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_20.cpu(),
            identities.cpu(),
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
            opt_in_prob=opt_in_prob,
            num_reserved_slots=num_reserved_slots,
        )
        self.assertTrue(torch.equal(output_readonly_cpu, output_readonly.cpu()))

        numbers_20_100 = torch.arange(20, 100, dtype=torch.int64, device="cuda")
        output_readonly, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_20_100,
            identities,
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
            opt_in_prob=opt_in_prob,
            num_reserved_slots=num_reserved_slots,
        )
        self.assertTrue(
            torch.all(
                (output_readonly >= num_opt_in_slots) & (output_readonly < zch_size)
            )
        )
        output_readonly_cpu, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_20_100.cpu(),
            identities.cpu(),
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
            opt_in_prob=opt_in_prob,
            num_reserved_slots=num_reserved_slots,
        )
        self.assertTrue(torch.equal(output_readonly_cpu, output_readonly.cpu()))

        # fill in all slots in the opt-in block and start eviction
        opt_in_rands = torch.full_like(numbers, 0, dtype=torch.int32, device="cuda")
        torch.ops.fbgemm.zero_collision_hash(
            numbers,
            identities,
            100,
            circular_probe=True,
            exp_hours=7 * 24,
            metadata=metadata,
            opt_in_prob=opt_in_prob,
            num_reserved_slots=num_reserved_slots,
            opt_in_rands=opt_in_rands,
        )
        identities_opt_in_slots = identities[:num_opt_in_slots]
        self.assertTrue(torch.all(identities_opt_in_slots != -1))

        metadata[:, 0] -= 7 * 24 + 1

        # number 101/102 are expected to be probed in opt-in/preserved blocks, respectively
        number_101_102 = torch.tensor([101, 102], dtype=torch.int64, device="cuda")
        opt_in_rands_101_102 = torch.tensor([10, 80], dtype=torch.int32, device="cuda")
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            number_101_102,
            identities,
            100,
            circular_probe=True,
            exp_hours=7 * 24,
            metadata=metadata,
            opt_in_prob=opt_in_prob,
            num_reserved_slots=num_reserved_slots,
            opt_in_rands=opt_in_rands_101_102,
        )
        self.assertTrue(output[0] < num_opt_in_slots)
        self.assertTrue(output[1] >= num_opt_in_slots)
        self.assertTrue(evict_slots.numel() == 1)
        self.assertTrue(
            evict_slots[0] < num_opt_in_slots
        )  # no eviction in reserved block

        output_readonly, _ = torch.ops.fbgemm.zero_collision_hash(
            number_101_102,
            identities,
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
            opt_in_prob=opt_in_prob,
            num_reserved_slots=num_reserved_slots,
        )
        self.assertTrue(torch.equal(output_readonly, output))
        output_readonly_cpu, _ = torch.ops.fbgemm.zero_collision_hash(
            number_101_102.cpu(),
            identities.cpu(),
            100,
            circular_probe=True,
            exp_hours=-1,
            readonly=True,
            opt_in_prob=opt_in_prob,
            num_reserved_slots=num_reserved_slots,
        )
        self.assertTrue(torch.equal(output_readonly_cpu, output_readonly.cpu()))

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_zch_lru_evict_train_eval(self) -> None:
        """
        Test the zero collision hash with LRU eviction policy.

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior,
                an assertion error will be raised.
        """
        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            100, support_evict=True, long_type=True, device=torch.device("cuda")
        )
        numbers_0_100 = torch.arange(0, 100, dtype=torch.int64, device="cuda")
        cur_hour = 1000
        ttl = 24
        input_metadata = torch.full_like(
            numbers_0_100,
            ttl + cur_hour,  # TTL 24h
            dtype=torch.int32,
            device="cuda",
        )
        output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
            numbers_0_100,
            identities,
            100,
            circular_probe=True,
            metadata=metadata,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
            input_metadata=input_metadata,
            eviction_threshold=cur_hour,
        )

        self.assertTrue(
            torch.equal(
                torch.sort(identities[identities != -1].view(-1))[0],
                numbers_0_100,
            ),
            f"{identities=}",
        )
        self.assertTrue(evict_slots.numel() == 0)

        identities_copy = identities.detach().clone()
        numbers_80_120 = torch.arange(80, 120, dtype=torch.int64, device="cuda")
        # gpu - readonly lookup: eval
        output_readonly, evictions = torch.ops.fbgemm.zero_collision_hash(
            numbers_80_120,
            identities,
            100,
            circular_probe=True,
            readonly=True,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
        )

        # check identities are not changed during readonly lookup
        self.assertTrue(
            torch.equal(identities_copy, identities),
            f"{identities_copy=} v.s {identities=}",
        )
        self.assertTrue(evictions is None)

        # [80, 100) will found at identities table, [100, 120) can't be found
        for idx in range(0, 20):
            self.assertEqual(
                identities[output_readonly[idx]],
                numbers_80_120[idx],
                f"{idx=}, {identities=}, {output_readonly=},  {numbers_80_120[idx]=}",
            )

        for idx in range(20, 40):
            self.assertNotEqual(
                identities[output_readonly[idx]],
                numbers_80_120[idx],
                f"{idx=}, {identities=}, {output_readonly=},  {numbers_80_120[idx]=}",
            )

        output_readonly_cpu, _ = torch.ops.fbgemm.zero_collision_hash(
            numbers_80_120.cpu(),
            identities.cpu(),
            100,
            circular_probe=True,
            readonly=True,
            eviction_policy=HashZchKernelEvictionPolicy.LRU_EVICTION.value,
        )
        self.assertTrue(
            torch.equal(output_readonly_cpu, output_readonly.cpu()),
            f"{output_readonly_cpu=} v.s {output_readonly=}",
        )

    @skipIfRocm("The CUDA kernel is not supported on ROCm")
    @unittest.skipIf(*gpu_unavailable)
    def test_murmur_hash(self) -> None:
        """
        This test is to verify the correctness of murmur hash3 kernel.
        Given a tensor, two rounds of murmur hash3 should return the same result.

        Assertions:
            1. output of two rounds of murmur hash3 should be equal

        Raises:
            AssertionError: If the test detects a mismatch from the expected behavior.
        """
        # # test on cpu
        input_item = torch.tensor([10000], dtype=torch.int64)
        output_item_first_round = torch.ops.fbgemm.murmur_hash3(input_item, 0, 0)
        output_item_second_round = torch.ops.fbgemm.murmur_hash3(input_item, 0, 0)
        self.assertTrue(torch.equal(output_item_first_round, output_item_second_round))
        # test on gpu
        input_item = torch.tensor([10000], dtype=torch.int64, device="cuda")
        output_item_first_round = torch.ops.fbgemm.murmur_hash3(input_item, 0, 0)
        output_item_second_round = torch.ops.fbgemm.murmur_hash3(input_item, 0, 0)
        self.assertTrue(torch.equal(output_item_first_round, output_item_second_round))
