#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import itertools
import random
import unittest

import hypothesis.strategies as st
import numpy as np
import torch
import torch._dynamo
from hypothesis import given, settings, Verbosity

from .common import additional_decorators, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, optests


def hash_size_cumsum_to_offsets(hash_size_cum_sum_list: list[int]) -> list[int]:
    hash_size_offsets_list = [0]
    count = 0
    for f in range(1, len(hash_size_cum_sum_list)):
        count = count + 1
        if hash_size_cum_sum_list[f] == hash_size_cum_sum_list[f - 1]:
            curr_offsets = hash_size_offsets_list[-1]
            hash_size_offsets_list.append(curr_offsets)
        else:
            hash_size_offsets_list.append(count)
    hash_size_offsets_list[-1] = count
    return hash_size_offsets_list


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class UniqueIndicesTest(unittest.TestCase):
    def setUp(self) -> None:
        assert hasattr(
            torch._dynamo.config, "assume_static_by_default"
        ), "Need to update the config as the dynamic/auto-dynamic setting has changed"
        # Turn off static assumption for auto-dynamic
        torch._dynamo.config.assume_static_by_default = False

    def _run_op(
        self,
        hash_size_cumsum_list: list[int],
        hash_size_offsets_list: list[int],
        lengths_list: list[int],
        indices_list: list[int],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build tensors from Python lists and invoke jagged_unique_indices."""
        device = torch.accelerator.current_accelerator()
        assert device is not None
        hash_size_cumsum = torch.as_tensor(
            hash_size_cumsum_list, dtype=dtype, device=device
        )
        hash_size_offsets = torch.as_tensor(
            hash_size_offsets_list, dtype=dtype, device=device
        )
        lengths = torch.as_tensor(lengths_list, dtype=dtype, device=device)
        indices = torch.as_tensor(indices_list, dtype=dtype, device=device)
        offsets = torch.zeros(len(lengths_list) + 1, dtype=dtype, device=device)
        offsets[1:] = torch.cumsum(lengths, dim=0)
        return torch.ops.fbgemm.jagged_unique_indices(
            hash_size_cumsum, hash_size_offsets, offsets, indices
        )

    def _check_sum_and_roundtrip(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        indices_list: list[int],
    ) -> None:
        """Standard cross-kernel invariants:
        (a) sum(output_lengths) accounts for every unique key.
        (b) unique_indices[reverse_index[i]] == indices[i] for every input i.
        """
        output_lengths, _, unique_indices, reverse_index = outputs
        self.assertEqual(int(torch.sum(output_lengths).item()), unique_indices.numel())
        rev_list = reverse_index.tolist()
        uniq_list = unique_indices.tolist()
        self.assertEqual(len(rev_list), len(indices_list))
        for i, rev in enumerate(rev_list):
            self.assertTrue(0 <= rev < len(uniq_list))
            self.assertEqual(uniq_list[rev], indices_list[i])

    @unittest.skipIf(*gpu_unavailable)
    @given(
        B=st.integers(min_value=100, max_value=200),
        F=st.integers(min_value=50, max_value=100),
        max_length=st.integers(min_value=5, max_value=10),
        dtype=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_jagged_unique_indices(
        self,
        B: int,  # Batch size
        F: int,  # The number of features
        max_length: int,  # The maximum value of pooling factor
        dtype: torch.dtype,
    ) -> None:
        hash_size_list = []
        lengths_list = []
        indices_list = []
        linearized_indices_list = []
        hash_size_offsets_list = [0]
        for _ in range(F):
            # We generate a small hash size to increase index duplication
            hash_size = random.randint(3, 5)
            hash_size_list.append(hash_size)
            hash_size_offset = hash_size_offsets_list[-1] + 1
            hash_size_offsets_list.append(hash_size_offset)
            for _ in range(B):
                length = random.randint(0, max_length)
                lengths_list.append(length)
                if length > 0:
                    indices = np.random.randint(0, hash_size, size=length)
                    linearized_indices = indices + sum(hash_size_list[:-1])
                    indices_list.extend(indices)
                    linearized_indices_list.extend(linearized_indices)

        device = torch.accelerator.current_accelerator()
        assert device is not None
        hash_size = torch.as_tensor(hash_size_list, dtype=dtype, device=device)
        hash_size_offsets = torch.as_tensor(
            hash_size_offsets_list, dtype=dtype, device=device
        )
        lengths = torch.as_tensor(lengths_list, dtype=dtype, device=device)
        indices = torch.as_tensor(indices_list, dtype=dtype, device=device)
        linearized_indices = torch.as_tensor(
            linearized_indices_list, dtype=dtype, device=device
        )

        hash_size_cum_sum = torch.zeros(F + 1, dtype=dtype, device=device)
        hash_size_cum_sum[1:] = torch.cumsum(hash_size, dim=0)
        offsets = torch.zeros(F * B + 1, dtype=dtype, device=device)
        offsets[1:] = torch.cumsum(lengths, dim=0)

        (
            output_lengths,
            output_offsets,
            unique_indices,
            reverse_index,
        ) = torch.ops.fbgemm.jagged_unique_indices(
            hash_size_cum_sum, hash_size_offsets, offsets, indices
        )

        # Check hash size cumsum to offsets function
        output_hash_size_offsets_list = hash_size_cumsum_to_offsets(
            hash_size_cum_sum.tolist()
        )
        self.assertEqual(output_hash_size_offsets_list, hash_size_offsets_list)

        # Compute hash size cumsum and offsets based on KJT offsets and indices
        (
            inferred_hash_size_cum_sum,
            inferred_hash_size_offsets,
        ) = torch.ops.fbgemm.jagged_hash_size_cumsum(offsets, indices, B)
        (
            output_lengths_inf,
            output_offsets_inf,
            unique_indices_inf,
            reverse_index_inf,
        ) = torch.ops.fbgemm.jagged_unique_indices(
            inferred_hash_size_cum_sum, inferred_hash_size_offsets, offsets, indices
        )

        self.assertTrue(torch.equal(output_lengths, output_lengths_inf))
        self.assertTrue(torch.equal(output_offsets, output_offsets_inf))
        self.assertTrue(torch.equal(unique_indices, unique_indices_inf))
        self.assertTrue(torch.equal(reverse_index, reverse_index_inf))

        unique_linearized_indices = torch.unique(linearized_indices, sorted=True)
        self.assertTrue(unique_linearized_indices.numel() == unique_indices.numel())

        unique_indices_list = unique_indices.tolist()
        reverse_index_list = reverse_index.tolist()
        for i in range(len(reverse_index_list)):
            pos = reverse_index_list[i]
            self.assertTrue(unique_indices_list[pos] == indices_list[i])

        input_offsets_list = offsets.tolist()
        output_offsets_list = output_offsets.tolist()
        for i in range(F):
            input_start = input_offsets_list[i * B]
            input_end = input_offsets_list[(i + 1) * B]
            output_start = output_offsets_list[i * B]
            output_end = output_offsets_list[(i + 1) * B]
            for each_offset in range(input_start, input_end):
                pos = reverse_index_list[each_offset]
                self.assertTrue((output_start <= pos) and (pos < output_end))

    @unittest.skipIf(*gpu_unavailable)
    @given(
        B=st.integers(min_value=100, max_value=200),
        F=st.integers(min_value=50, max_value=100),
        max_length=st.integers(min_value=5, max_value=10),
        dtype=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_jagged_unique_indices_multi_keys(
        self,
        B: int,  # Batch size
        F: int,  # The number of features
        max_length: int,  # The maximum value of pooling factor
        dtype: torch.dtype,
    ) -> None:
        hash_size_list = []
        lengths_list = []
        indices_list = []
        linearized_indices_list = []
        MAX_HASH_SIZE = 10
        for _ in range(F):
            # We generate a small hash size to increase index duplication
            hash_size = random.randint(3, 6)
            self.assertTrue(hash_size <= MAX_HASH_SIZE)
            masked_hash_size = MAX_HASH_SIZE if random.randint(1, 3) == 3 else 0
            hash_size_list.append(masked_hash_size)
            for _ in range(B):
                length = random.randint(0, max_length)
                lengths_list.append(length)
                if length > 0:
                    indices = np.random.randint(0, hash_size, size=length)
                    linearized_indices = indices + sum(hash_size_list[:-1])
                    indices_list.extend(indices)
                    linearized_indices_list.extend(linearized_indices)

        device = torch.accelerator.current_accelerator()
        assert device is not None
        hash_size = torch.as_tensor(hash_size_list, dtype=dtype, device=device)
        lengths = torch.as_tensor(lengths_list, dtype=dtype, device=device)
        indices = torch.as_tensor(indices_list, dtype=dtype, device=device)
        linearized_indices = torch.as_tensor(
            linearized_indices_list, dtype=dtype, device=device
        )

        hash_size_cum_sum = torch.zeros(F + 1, dtype=dtype, device=device)
        hash_size_cum_sum[1:] = torch.cumsum(hash_size, dim=0)
        offsets = torch.zeros(F * B + 1, dtype=dtype, device=device)
        offsets[1:] = torch.cumsum(lengths, dim=0)

        # Compute hash size offsets based on hash size cumsum to dedup
        # indices from multiple keys
        hash_size_cum_sum_list = hash_size_cum_sum.tolist()
        hash_size_offsets_list = hash_size_cumsum_to_offsets(hash_size_cum_sum_list)
        hash_size_offsets = torch.as_tensor(
            hash_size_offsets_list, dtype=dtype, device=device
        )

        (
            _,  # output lengths
            _,  # output offsets
            unique_indices,
            reverse_index,
        ) = torch.ops.fbgemm.jagged_unique_indices(
            hash_size_cum_sum, hash_size_offsets, offsets, indices
        )

        unique_linearized_indices = torch.unique(linearized_indices, sorted=True)
        self.assertTrue(unique_linearized_indices.numel() == unique_indices.numel())

        unique_indices_list = unique_indices.tolist()
        reverse_index_list = reverse_index.tolist()
        for i in range(len(reverse_index_list)):
            pos = reverse_index_list[i]
            self.assertTrue(unique_indices_list[pos] == indices_list[i])

    @unittest.skipIf(*gpu_unavailable)
    @given(
        B=st.integers(min_value=100, max_value=200),
        F=st.integers(min_value=50, max_value=100),
        dtype=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_jagged_unique_indices_empty(
        self,
        B: int,  # Batch size
        F: int,  # The number of features
        dtype: torch.dtype,
    ) -> None:
        hash_size_cumsum_list = [0] + list(itertools.accumulate([10] * F))
        hash_size_offsets_list = [0] + list(itertools.accumulate([1] * F))
        offsets_list = [0] * (B * F + 1)
        indices_list = []

        device = torch.accelerator.current_accelerator()
        assert device is not None
        hash_size_cumsum = torch.as_tensor(
            hash_size_cumsum_list, device=device, dtype=dtype
        )
        hash_size_offsets = torch.as_tensor(
            hash_size_offsets_list, device=device, dtype=dtype
        )
        offsets = torch.as_tensor(offsets_list, device=device, dtype=dtype)
        indices = torch.as_tensor(indices_list, device=device, dtype=dtype)

        (
            output_lengths,
            output_offsets,
            unique_indices,
            reverse_index,
        ) = torch.ops.fbgemm.jagged_unique_indices(
            hash_size_cumsum, hash_size_offsets, offsets, indices
        )

        # The output should be empty since there are no input indices
        self.assertEqual(unique_indices.numel(), 0)
        self.assertEqual(reverse_index.numel(), 0)
        self.assertEqual(torch.sum(output_lengths).item(), 0)
        self.assertEqual(torch.sum(output_offsets).item(), 0)

    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_unique_indices_oob_last_feature(self) -> None:
        """Exercise the op when the last feature has indices exceeding its
        per-feature hash_size. The linearized form
        `hash_size_cumsum[T-1] + idx` then exceeds hash_size_cumsum[T], which
        sorts past the length kernel's upper binary-search boundary.
        Regressing here causes sum(output_lengths) < unique_indices.numel(),
        producing an inconsistent KJT that crashes downstream
        all_to_all_single with "Split sizes doesn't match total dim 0 size".

        Also asserts the inverse-index round-trip
        unique_indices[reverse_index[i]] == indices[i] for the OOB values,
        which fails for a gather-based delinearize that derives the
        per-feature index from the linearized key.

        Sweeps several T values so the carve-out is exercised both at the
        T=2 boundary and with intermediate in-bound features preceding the
        OOB last feature, and both index dtypes (int32, int64) so the
        templated kernel paths are both covered.
        """
        for T, dtype in itertools.product((2, 3, 5, 8), (torch.int32, torch.int64)):
            with self.subTest(T=T, dtype=dtype):
                self._run_oob_last_feature(T, dtype)

    def _run_oob_last_feature(self, T: int, dtype: torch.dtype) -> None:
        random.seed(T)
        np.random.seed(T)
        B = 64
        max_length = 5
        per_feature_hash_size = 100
        # Last feature's OOB indices land far above
        # hash_size_cumsum[T]=T*per_feature_hash_size.
        oob_value_cap = 100_000
        hash_size_cumsum_list = [t * per_feature_hash_size for t in range(T + 1)]
        hash_size_offsets_list = list(range(T + 1))
        lengths_list: list[int] = []
        indices_list: list[int] = []
        for t in range(T):
            value_cap = oob_value_cap if t == T - 1 else per_feature_hash_size
            for _ in range(B):
                length = random.randint(0, max_length)
                lengths_list.append(length)
                if length > 0:
                    indices_list.extend(
                        np.random.randint(0, value_cap, size=length).tolist()
                    )
        # Force at least one OOB value on the last feature so the test
        # exercises the carve-out even if the random draw happened to stay
        # in range.
        if lengths_list[-1] == 0:
            lengths_list[-1] = 1
            indices_list.append(oob_value_cap - 1)
        else:
            indices_list[-1] = oob_value_cap - 1

        outputs = self._run_op(
            hash_size_cumsum_list,
            hash_size_offsets_list,
            lengths_list,
            indices_list,
            dtype,
        )
        self._check_sum_and_roundtrip(outputs, indices_list)

    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_unique_indices_oob_duplicates(self) -> None:
        """Identical OOB values on the last feature must collapse to a
        single unique slot. The scatter form of delinearize has multiple
        threads write to the same unique_indices[reverse_index[i]];
        correctness relies on every writer for a given slot writing the
        same value (true here because the inputs are literal duplicates).
        """
        B = 32
        per_hash = 100
        oob_value = 99_999
        hash_size_cumsum_list = [0, per_hash, 2 * per_hash]
        hash_size_offsets_list = [0, 1, 2]
        # Feature 0: one distinct in-bound value per batch.
        # Feature 1: B identical copies of oob_value across batches.
        lengths_list = [1] * (2 * B)
        indices_list: list[int] = list(range(B)) + [oob_value] * B
        outputs = self._run_op(
            hash_size_cumsum_list,
            hash_size_offsets_list,
            lengths_list,
            indices_list,
            torch.int64,
        )
        self._check_sum_and_roundtrip(outputs, indices_list)
        _, _, unique_indices, reverse_index = outputs
        # B distinct in-bound values + 1 OOB unique slot.
        self.assertEqual(unique_indices.numel(), B + 1)
        last_feature_revs = reverse_index[B : 2 * B].tolist()
        self.assertEqual(len(set(last_feature_revs)), 1)
        self.assertEqual(unique_indices[last_feature_revs[0]].item(), oob_value)

    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_unique_indices_oob_all_values_last_feature(self) -> None:
        """Every value on the last feature is strictly OOB
        (idx >= per_feature_hash_size, no in-bound mix). Exercises the
        boundary where lo_pos lands past any in-bound values for the last
        group, and the clamp captures the full OOB tail.
        """
        random.seed(0)
        np.random.seed(0)
        B = 32
        per_hash = 100
        hash_size_cumsum_list = [0, per_hash, 2 * per_hash]
        hash_size_offsets_list = [0, 1, 2]
        lengths_list: list[int] = []
        indices_list: list[int] = []
        for _ in range(B):  # feature 0: in-bound
            length = random.randint(1, 4)
            lengths_list.append(length)
            indices_list.extend(np.random.randint(0, per_hash, size=length).tolist())
        for _ in range(B):  # feature 1: strictly OOB (>= per_hash)
            length = random.randint(1, 4)
            lengths_list.append(length)
            indices_list.extend(
                np.random.randint(per_hash, per_hash * 1000, size=length).tolist()
            )
        outputs = self._run_op(
            hash_size_cumsum_list,
            hash_size_offsets_list,
            lengths_list,
            indices_list,
            torch.int64,
        )
        self._check_sum_and_roundtrip(outputs, indices_list)

    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_unique_indices_oob_size_zero_last_feature(self) -> None:
        """Last feature has per_feature_hash_size == 0 (consecutive equal
        cumsum entries at the tail); any positive value on it is OOB by
        construction. The length kernel's `low == high` shortcut must not
        drop the OOB tail for the last group.
        """
        random.seed(0)
        np.random.seed(0)
        B = 32
        per_hash = 100
        hash_size_cumsum_list = [0, per_hash, per_hash]
        hash_size_offsets_list = [0, 1, 2]
        lengths_list: list[int] = []
        indices_list: list[int] = []
        for _ in range(B):  # feature 0: in-bound
            length = random.randint(1, 4)
            lengths_list.append(length)
            indices_list.extend(np.random.randint(0, per_hash, size=length).tolist())
        for _ in range(B):  # feature 1: all values OOB (last feature, size 0)
            length = random.randint(1, 4)
            lengths_list.append(length)
            indices_list.extend(np.random.randint(1, 10_000, size=length).tolist())
        outputs = self._run_op(
            hash_size_cumsum_list,
            hash_size_offsets_list,
            lengths_list,
            indices_list,
            torch.int64,
        )
        self._check_sum_and_roundtrip(outputs, indices_list)

    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_unique_indices_oob_attributed_to_last_group(self) -> None:
        """The OOB tail must be counted into the last feature group's
        output_lengths slice, not an earlier one. A regression that
        applied the clamp to non-last groups would still pass the sum
        invariant but would mis-attribute the tail to the wrong feature.
        Earlier groups can hold at most `per_feature_hash_size` unique
        values each; the rest must land in the last group.
        """
        random.seed(0)
        np.random.seed(0)
        T = 3
        B = 16
        per_hash = 100
        hash_size_cumsum_list = [t * per_hash for t in range(T + 1)]
        hash_size_offsets_list = list(range(T + 1))
        lengths_list: list[int] = []
        indices_list: list[int] = []
        for t in range(T):
            cap = 100_000 if t == T - 1 else per_hash
            for _ in range(B):
                length = random.randint(1, 4)
                lengths_list.append(length)
                indices_list.extend(np.random.randint(0, cap, size=length).tolist())
        outputs = self._run_op(
            hash_size_cumsum_list,
            hash_size_offsets_list,
            lengths_list,
            indices_list,
            torch.int64,
        )
        self._check_sum_and_roundtrip(outputs, indices_list)
        output_lengths = outputs[0]
        for t in range(T - 1):
            group_total = int(torch.sum(output_lengths[t * B : (t + 1) * B]).item())
            self.assertLessEqual(group_total, per_hash)

    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_unique_indices_oob_low_bit_collision_last_feature(
        self,
    ) -> None:
        """A last-feature OOB key whose low bits collide with an in-bound key
        in an earlier feature must still be ordered by its full value.

        The radix-sort end_bit is derived from the maximum linearized key. If
        it were instead trimmed to bit_width(total_hash_size), cub would
        ignore the OOB key's high bits and sort only by the low bits. Here
        total_hash_size = 200 (bit_width 8, i.e. the low byte). The last
        feature's hash_offset is 100, so the OOB value 180 linearizes to
        100 + 180 = 280, and 280 % 256 == 24 -- smaller than the in-bound
        feature-0 key 50. A low-byte-only sort would place the OOB key
        *before* the in-bound key, so the length kernel's binary searches
        mis-attribute the OOB unique to feature 0: output_lengths becomes
        [2, 0] instead of [1, 1]. sum(output_lengths) == num_unique
        telescopes either way, so only the exact per-feature counts catch
        the regression. Swept over both index dtypes.
        """
        per_hash = 100  # total_hash_size = 200 -> trimmed end_bit would be 8
        inbound_feature0 = 50
        oob_last_feature = 180  # 100 + 180 = 280; 280 % 256 == 24 < 50
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                outputs = self._run_op(
                    hash_size_cumsum_list=[0, per_hash, 2 * per_hash],
                    hash_size_offsets_list=[0, 1, 2],
                    lengths_list=[1, 1],  # one sample per feature
                    indices_list=[inbound_feature0, oob_last_feature],
                    dtype=dtype,
                )
                self._check_sum_and_roundtrip(
                    outputs, [inbound_feature0, oob_last_feature]
                )
                output_lengths, _, unique_indices, _ = outputs
                # Exactly one unique per feature, attributed correctly.
                self.assertEqual(output_lengths.tolist(), [1, 1])
                self.assertCountEqual(
                    unique_indices.tolist(),
                    [inbound_feature0, oob_last_feature],
                )

    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_unique_indices_oob_low_bit_collision_overcount(self) -> None:
        """Duplicate OOB keys separated by a low-bit-colliding distinct key
        must still dedup to a single unique slot.

        The adjacent-diff + RLE dedup compares full key values, so it only
        collapses duplicates that the radix sort placed adjacently. With
        total_hash_size = 200 (trimmed end_bit 8), the last feature's keys
        100 + {180, 436, 180} = {280, 536, 280} all share low byte 24, and a
        low-byte-only sort preserves their input order [280, 536, 280] -- so
        the two copies of 280 are split by 536 and counted as two distinct
        uniques, over-counting num_unique (6 instead of 5). Deriving end_bit
        from the real max key (536, bit_width 10) sorts them to [280, 280,
        536] and the dedup collapses correctly.
        """
        per_hash = 100  # total_hash_size = 200 -> trimmed end_bit would be 8
        # Feature 0: three distinct in-bound values (low bytes 10/11/12).
        # Feature 1 (last): 100 + {180, 436, 180} = {280, 536, 280}, all low
        # byte 24; 180 appears twice and must collapse to one unique.
        feature0 = [10, 11, 12]
        last_feature = [180, 436, 180]
        outputs = self._run_op(
            hash_size_cumsum_list=[0, per_hash, 2 * per_hash],
            hash_size_offsets_list=[0, 1, 2],
            lengths_list=[1] * 6,  # batch_size 3 per feature
            indices_list=feature0 + last_feature,
            dtype=torch.int64,
        )
        self._check_sum_and_roundtrip(outputs, feature0 + last_feature)
        _, _, unique_indices, _ = outputs
        # True unique set: {10, 11, 12} + {180, 436} = 5 distinct values.
        self.assertEqual(unique_indices.numel(), 5)
        self.assertCountEqual(set(unique_indices.tolist()), {10, 11, 12, 180, 436})

    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_unique_indices_merged_intermediate_oob(self) -> None:
        """Intermediate feature with per_feature_hash == 0 is part of a
        merged group (consecutive equal entries in hash_size_cumsum). The
        contract assert exempts these; index values up to the shared
        group's total hash space are valid. Verify the op handles the
        merged group end-to-end: round-trip preserves the original
        per-feature value, and the merged group's total length equals the
        merged unique count.
        """
        random.seed(0)
        np.random.seed(0)
        B = 16
        # Feature 1 merged with feature 2: shared hash space [100, 200).
        # hash_size_offsets maps groups [0, 1) and [1, 3) -> features {0}
        # and {1, 2} respectively.
        hash_size_cumsum_list = [0, 100, 100, 200]
        hash_size_offsets_list = [0, 1, 1, 3]
        lengths_list: list[int] = []
        indices_list: list[int] = []
        for _ in range(B):  # feature 0: in-bound [0, 100)
            length = random.randint(1, 4)
            lengths_list.append(length)
            indices_list.extend(np.random.randint(0, 100, size=length).tolist())
        for _ in range(B):  # feature 1: merged intermediate, [0, 100)
            length = random.randint(1, 4)
            lengths_list.append(length)
            indices_list.extend(np.random.randint(0, 100, size=length).tolist())
        for _ in range(B):  # feature 2: last in merged group, [0, 100)
            length = random.randint(1, 4)
            lengths_list.append(length)
            indices_list.extend(np.random.randint(0, 100, size=length).tolist())
        outputs = self._run_op(
            hash_size_cumsum_list,
            hash_size_offsets_list,
            lengths_list,
            indices_list,
            torch.int64,
        )
        self._check_sum_and_roundtrip(outputs, indices_list)

    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_unique_indices_zch_huge_hash_size(self) -> None:
        """Exercise the op with a hash_size_cumsum entry at INT64_MAX -
        the shape produced by ZCH callers that leave per-feature hash size
        unbounded. The op must handle hash boundaries spanning the full
        int64 range without overflow in any internal arithmetic.
        """
        T = 2
        B = 64
        max_length = 5
        int64_max = torch.iinfo(torch.int64).max
        hash_size_cumsum_list = [0, 0, int64_max]
        hash_size_offsets_list = [0, 0, 2]
        # Per-feature linearized values lie in [0, INT64_MAX). The kernels
        # under test are boundary-value-sensitive on hash_size_cumsum, not
        # on the indices themselves, so a small index range is sufficient
        # and keeps the reference comparison fast.
        per_feature_value_cap = 1024
        lengths_list: list[int] = []
        indices_list: list[int] = []
        for _ in range(T):
            for _ in range(B):
                length = random.randint(0, max_length)
                lengths_list.append(length)
                if length > 0:
                    indices_list.extend(
                        np.random.randint(
                            0, per_feature_value_cap, size=length
                        ).tolist()
                    )

        device = torch.accelerator.current_accelerator()
        assert device is not None
        dtype = torch.int64
        hash_size_cumsum = torch.as_tensor(
            hash_size_cumsum_list, dtype=dtype, device=device
        )
        hash_size_offsets = torch.as_tensor(
            hash_size_offsets_list, dtype=dtype, device=device
        )
        lengths = torch.as_tensor(lengths_list, dtype=dtype, device=device)
        indices = torch.as_tensor(indices_list, dtype=dtype, device=device)
        offsets = torch.zeros(T * B + 1, dtype=dtype, device=device)
        offsets[1:] = torch.cumsum(lengths, dim=0)

        (
            output_lengths,
            output_offsets,
            unique_indices,
            reverse_index,
        ) = torch.ops.fbgemm.jagged_unique_indices(
            hash_size_cumsum, hash_size_offsets, offsets, indices
        )

        # Both features share the same hash space (hash_offset = 0 for
        # both, since hash_size_cumsum[0] == hash_size_cumsum[1] == 0),
        # so the global unique set is the union of all input indices.
        expected_unique = sorted(set(indices_list))
        self.assertEqual(unique_indices.numel(), len(expected_unique))
        self.assertEqual(int(torch.sum(output_lengths).item()), unique_indices.numel())
        # Inverse-index round-trip: unique_indices[reverse_index[i]] == indices[i].
        rev_list = reverse_index.tolist()
        uniq_list = unique_indices.tolist()
        self.assertEqual(len(rev_list), len(indices_list))
        for i, rev in enumerate(rev_list):
            self.assertTrue(0 <= rev < len(uniq_list))
            self.assertEqual(uniq_list[rev], indices_list[i])

    @given(
        num_elements=st.integers(min_value=100, max_value=10000),
        num_unique_indices=st.integers(min_value=5, max_value=100),
        weight_dtype=st.sampled_from([torch.float32, torch.float16]),
        use_cpu=st.booleans() if not gpu_unavailable[0] else st.just(True),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_acc_weights_and_counts_1d(
        self,
        num_elements: int,
        num_unique_indices: int,
        weight_dtype: torch.dtype,
        use_cpu: bool,
    ) -> None:
        """Test 1D weight accumulation kernel against torch native implementation.

        Tests both CPU and GPU implementations.
        """
        device = torch.device("cpu" if use_cpu else "cuda")

        # Generate test data
        weights = torch.randn(num_elements, dtype=weight_dtype, device=device)
        reverse_indices = torch.randint(
            0, num_unique_indices, (num_elements,), dtype=torch.int64, device=device
        )

        # Test our optimized kernel
        result_optimized = torch.ops.fbgemm.jagged_acc_weights_and_counts(
            weights, reverse_indices, num_unique_indices
        )

        # Reference implementation using torch native operations
        result_reference = torch.zeros(
            (num_unique_indices, 2), dtype=torch.float32, device=device
        )

        # Accumulate weights and counts using scatter_add (torch native)
        weights_float = weights.float()
        counts = torch.ones_like(weights_float)

        result_reference[:, 0].scatter_add_(0, reverse_indices, weights_float)
        result_reference[:, 1].scatter_add_(0, reverse_indices, counts)

        # Compare results
        torch.testing.assert_close(
            result_optimized, result_reference, rtol=1e-4, atol=1e-5
        )

        # Verify output shape and types
        self.assertEqual(result_optimized.shape, (num_unique_indices, 2))
        self.assertEqual(result_optimized.dtype, torch.float32)
        self.assertEqual(result_optimized.device.type, device.type)

    @given(
        use_cpu=st.booleans() if not gpu_unavailable[0] else st.just(True),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_jagged_acc_weights_and_counts_edge_cases(self, use_cpu: bool) -> None:
        """Test edge cases for both 1D and 2D accumulation kernels.

        Tests both CPU and GPU implementations.
        """
        device = torch.device("cpu" if use_cpu else "cuda")

        # Test case 1: Single element
        weights_1d = torch.tensor([5.0], device=device)
        reverse_indices = torch.tensor([0], dtype=torch.int64, device=device)
        result = torch.ops.fbgemm.jagged_acc_weights_and_counts(
            weights_1d, reverse_indices, 1
        )
        expected = torch.tensor([[5.0, 1.0]], device=device)
        torch.testing.assert_close(result, expected)

        # Test case 2: All elements map to same unique index
        weights_1d = torch.tensor([1.0, 2.0, 3.0], device=device)
        reverse_indices = torch.tensor([0, 0, 0], dtype=torch.int64, device=device)
        result = torch.ops.fbgemm.jagged_acc_weights_and_counts(
            weights_1d, reverse_indices, 1
        )
        expected = torch.tensor([[6.0, 3.0]], device=device)
        torch.testing.assert_close(result, expected)

    @given(use_cpu=st.booleans() if not gpu_unavailable[0] else st.just(True))
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_jagged_acc_weights_and_counts_different_sizes(self, use_cpu: bool) -> None:
        """Test that the kernel works correctly with different dataset sizes.

        Tests both small and large datasets to ensure the implementation works
        correctly across different scales. For CPU, just tests basic functionality.
        """
        device = torch.device("cpu" if use_cpu else "cuda")

        # Test small dataset
        small_weights = torch.randn(500, device=device)
        small_reverse_indices = torch.randint(
            0, 10, (500,), dtype=torch.int64, device=device
        )
        result_small = torch.ops.fbgemm.jagged_acc_weights_and_counts(
            small_weights, small_reverse_indices, 10
        )

        # Test large dataset
        large_weights = torch.randn(5000, device=device)
        large_reverse_indices = torch.randint(
            0, 50, (5000,), dtype=torch.int64, device=device
        )
        result_large = torch.ops.fbgemm.jagged_acc_weights_and_counts(
            large_weights, large_reverse_indices, 50
        )

        # Both should produce valid results
        self.assertEqual(result_small.shape, (10, 2))
        self.assertEqual(result_large.shape, (50, 2))

        # Verify results are reasonable (non-negative counts, finite weights)
        self.assertTrue(
            torch.all(result_small[:, 1] >= 0)
        )  # Counts should be non-negative
        self.assertTrue(
            torch.all(result_large[:, 1] >= 0)
        )  # Counts should be non-negative
        self.assertTrue(torch.all(torch.isfinite(result_small)))
        self.assertTrue(torch.all(torch.isfinite(result_large)))


if __name__ == "__main__":
    unittest.main()
