# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import random
import unittest
from collections.abc import Callable

import fbgemm_gpu.tbe.ssd  # noqa F401
import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

from .. import common  # noqa E402
from ..common import gpu_unavailable, running_in_oss

MAX_EXAMPLES = 20


# pyrefly: ignore [bad-argument-type]
@unittest.skipIf(*running_in_oss)
# pyrefly: ignore [bad-argument-type]
@unittest.skipIf(*gpu_unavailable)
class SSDUtilsTest(unittest.TestCase):
    def execute_masked_index_test(
        self,
        D: int,
        max_index: int,
        num_indices: int,
        num_value_rows: int,
        num_output_rows: int,
        dtype: torch.dtype,
        test_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor
        ],
        is_index_put: bool,
        use_pipeline: bool,
    ) -> None:
        """
        A helper function that generates inputs/outputs, runs
        torch.ops.fbgemm.masked_index_* against the PyTorch counterpart, and
        compares the output results"""
        device = "cuda"

        # Number of columns must be multiple of 4 (embedding requirement)
        D = D * 4

        # Generate indices
        indices = torch.randint(
            low=0, high=max_index, size=(num_indices,), dtype=torch.long, device=device
        )

        # Compute/set unique indices (indices have to be unique to avoid race
        # condition)
        indices_unique = indices.unique()
        count_val = indices_unique.numel()
        indices[:count_val] = indices_unique

        # Permute unique indices
        rand_pos = torch.randperm(indices_unique.numel(), device=device)
        indices[:count_val] = indices[rand_pos]

        # Set some indices to -1
        indices[rand_pos[: max(count_val // 2, 1)]] = -1

        # Generate count tensor
        count = torch.as_tensor([count_val], dtype=torch.int, device=device)

        # Generate values
        values = torch.rand(num_value_rows, D, dtype=dtype, device=device)

        # Allocate output and output_ref
        output = torch.zeros(num_output_rows, D, dtype=dtype, device=device)
        output_ref = torch.zeros(num_output_rows, D, dtype=dtype, device=device)

        # Run test
        output = test_fn(output, indices, values, count, use_pipeline)

        # Run reference
        indices = indices[:count_val]
        filter_ = indices >= 0
        indices_ = indices[filter_]
        filter_locs = filter_.nonzero().flatten()
        if is_index_put:
            output_ref[indices_] = values[filter_locs]
        else:
            output_ref[filter_locs] = values[indices_]

        # Compare results
        assert torch.equal(output_ref, output)

    # pyre-ignore [56]
    @given(
        num_indices=st.integers(min_value=10, max_value=100),
        D=st.integers(min_value=2, max_value=256),
        num_output_rows=st.integers(min_value=10, max_value=100),
        dtype=st.sampled_from([torch.float, torch.half]),
        use_pipeline=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_masked_index_put(
        self,
        num_indices: int,
        D: int,
        num_output_rows: int,
        dtype: torch.dtype,
        use_pipeline: bool,
    ) -> None:
        """
        Test correctness of torch.ops.fbgemm.masked_index_put against PyTorch's
        index_put
        """
        self.execute_masked_index_test(
            D=D,
            max_index=num_output_rows,
            num_indices=num_indices,
            num_value_rows=num_indices,
            num_output_rows=num_output_rows,
            dtype=dtype,
            # pyre-fixme[6]: For 7th argument expected `(Tensor, Tensor, Tensor,
            #  Tensor, bool) -> Tensor` but got `OpOverloadPacket`.
            test_fn=torch.ops.fbgemm.masked_index_put,
            is_index_put=True,
            use_pipeline=use_pipeline,
        )

    # pyre-ignore [56]
    @given(
        num_indices=st.integers(min_value=10, max_value=100),
        D=st.integers(min_value=2, max_value=256),
        num_value_rows=st.integers(min_value=10, max_value=100),
        dtype=st.sampled_from([torch.float, torch.half]),
        use_pipeline=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_masked_index_select(
        self,
        num_indices: int,
        D: int,
        num_value_rows: int,
        dtype: torch.dtype,
        use_pipeline: bool,
    ) -> None:
        """
        Test correctness of torch.ops.fbgemm.masked_index_select aginst
        PyTorch's index_select
        """
        self.execute_masked_index_test(
            D=D,
            max_index=num_value_rows,
            num_indices=num_indices,
            num_value_rows=num_value_rows,
            num_output_rows=num_indices,
            dtype=dtype,
            # pyre-fixme[6]: For 7th argument expected `(Tensor, Tensor, Tensor,
            #  Tensor, bool) -> Tensor` but got `OpOverloadPacket`.
            test_fn=torch.ops.fbgemm.masked_index_select,
            is_index_put=False,
            use_pipeline=use_pipeline,
        )

    def test_kvzch_direct_cache_lookup(self) -> None:
        device = torch.accelerator.current_accelerator()
        embedding_dim = 4
        cache_sets = 4
        first_id_by_slot: dict[int, int] = {}
        collision_keys: list[int] = []
        for key in range(100):
            slot = torch.ops.fbgemm.lxu_cache_slot(key, cache_sets)
            if slot in first_id_by_slot:
                collision_keys = [first_id_by_slot[slot], key]
                break
            first_id_by_slot[slot] = key
        self.assertEqual(len(collision_keys), 2)
        hit_key, miss_key = collision_keys
        slot = torch.ops.fbgemm.lxu_cache_slot(hit_key, cache_sets)
        cache_state = torch.full((cache_sets, 1), -1, dtype=torch.int64, device=device)
        cache_state[slot][0] = hit_key
        cache_weights = torch.zeros(
            (cache_sets, embedding_dim), dtype=torch.float32, device=device
        )
        cache_weights[slot] = torch.arange(
            1, embedding_dim + 1, dtype=torch.float32, device=device
        )
        indices = torch.tensor(
            [hit_key, miss_key, hit_key], dtype=torch.int64, device=device
        )
        action_indices = torch.empty_like(indices)
        actions_count = torch.zeros(1, dtype=torch.int32, device=device)
        inserted_rows = torch.zeros(
            (indices.numel(), embedding_dim), dtype=torch.float32, device=device
        )
        row_addresses = torch.empty_like(indices)

        torch.ops.fbgemm.kvzch_direct_cache_lookup(
            indices,
            cache_state,
            cache_weights,
            action_indices,
            actions_count,
            inserted_rows,
            row_addresses,
        )
        self.assertEqual(int(actions_count.item()), 1)
        self.assertEqual(int(action_indices[0].item()), miss_key)
        miss_row = torch.arange(
            11, 11 + embedding_dim, dtype=torch.float32, device=device
        )
        inserted_rows[0].copy_(miss_row)
        output = torch.empty_like(inserted_rows)
        torch.ops.fbgemm.ssd_gather_from_row_addresses(output, row_addresses)

        hit_row = cache_weights[slot]
        torch.testing.assert_close(
            output,
            torch.stack((hit_row, miss_row, hit_row)),
        )

    def test_kvzch_direct_cache_insert(self) -> None:
        device = torch.accelerator.current_accelerator()
        embedding_dim = 4
        cache_sets = 4
        first_id_by_slot: dict[int, int] = {}
        collision_keys: list[int] = []
        for key in range(100):
            slot = torch.ops.fbgemm.lxu_cache_slot(key, cache_sets)
            if slot in first_id_by_slot:
                collision_keys = [first_id_by_slot[slot], key]
                break
            first_id_by_slot[slot] = key
        self.assertEqual(len(collision_keys), 2)
        first_key, replacement_key = collision_keys
        slot = torch.ops.fbgemm.lxu_cache_slot(first_key, cache_sets)
        cache_state = torch.full((cache_sets, 1), -1, dtype=torch.int64, device=device)
        cache_weights = torch.zeros(
            (cache_sets, embedding_dim), dtype=torch.float32, device=device
        )
        first_row = torch.arange(
            1, embedding_dim + 1, dtype=torch.float32, device=device
        ).view(1, -1)

        torch.ops.fbgemm.kvzch_direct_cache_insert(
            cache_state,
            cache_weights,
            torch.tensor([first_key], dtype=torch.int64, device=device),
            first_row,
        )
        self.assertEqual(int(cache_state[slot][0].item()), first_key)
        torch.testing.assert_close(cache_weights[slot], first_row[0])

        zero_key = next(
            key
            for key in range(100, 200)
            if torch.ops.fbgemm.lxu_cache_slot(key, cache_sets) != slot
        )
        zero_slot = torch.ops.fbgemm.lxu_cache_slot(zero_key, cache_sets)
        torch.ops.fbgemm.kvzch_direct_cache_insert(
            cache_state,
            cache_weights,
            torch.tensor([zero_key], dtype=torch.int64, device=device),
            torch.zeros_like(first_row),
        )
        self.assertEqual(int(cache_state[zero_slot][0].item()), -1)

        cache_state.fill_(-1)
        cache_weights.zero_()
        prefix_count = torch.tensor([1], dtype=torch.int32, device=device)
        torch.ops.fbgemm.kvzch_direct_cache_insert(
            cache_state,
            cache_weights,
            torch.tensor([first_key, zero_key], dtype=torch.int64, device=device),
            torch.cat((first_row, first_row + 20)),
            prefix_count,
        )
        self.assertEqual(int(cache_state[slot][0].item()), first_key)
        self.assertEqual(int(cache_state[zero_slot][0].item()), -1)
        torch.testing.assert_close(cache_weights[slot], first_row[0])

        replacement_row = first_row + 10
        torch.ops.fbgemm.kvzch_direct_cache_insert(
            cache_state,
            cache_weights,
            torch.tensor([replacement_key], dtype=torch.int64, device=device),
            replacement_row,
        )
        self.assertEqual(int(cache_state[slot][0].item()), replacement_key)
        torch.testing.assert_close(cache_weights[slot], replacement_row[0])

        cache_state.fill_(-1)
        cache_weights.zero_()
        collision_rows = torch.cat((first_row, replacement_row))
        torch.ops.fbgemm.kvzch_direct_cache_insert(
            cache_state,
            cache_weights,
            torch.tensor(collision_keys, dtype=torch.int64, device=device),
            collision_rows,
        )
        winning_key = int(cache_state[slot][0].item())
        self.assertIn(winning_key, collision_keys)
        winning_row = collision_rows[collision_keys.index(winning_key)]
        torch.testing.assert_close(cache_weights[slot], winning_row)

    @staticmethod
    def _unique_slot_keys(cache_sets: int, num_keys: int, start: int = 0) -> list[int]:
        used_slots: set[int] = set()
        keys: list[int] = []
        key = start
        while len(keys) < num_keys:
            slot = torch.ops.fbgemm.lxu_cache_slot(key, cache_sets)
            if slot not in used_slots:
                used_slots.add(slot)
                keys.append(key)
            key += 1
        return keys

    def _run_direct_cache_lookup(
        self,
        op: Callable[..., torch.Tensor],
        indices: torch.Tensor,
        cache_state: torch.Tensor,
        cache_weights: torch.Tensor,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_indices = torch.full_like(indices, -7)
        actions_count = torch.zeros(1, dtype=torch.int32, device=indices.device)
        inserted_rows = torch.zeros(
            (indices.numel(), embedding_dim), dtype=dtype, device=indices.device
        )
        row_addresses = torch.empty_like(indices)
        op(
            indices,
            cache_state,
            cache_weights,
            action_indices,
            actions_count,
            inserted_rows,
            row_addresses,
        )
        return action_indices, actions_count, inserted_rows, row_addresses

    def test_kvzch_direct_cache_int8_roundtrip(self) -> None:
        device = torch.accelerator.current_accelerator()
        embedding_dim = 128
        cache_sets = 64
        num_resident = 16
        keys = self._unique_slot_keys(cache_sets, num_resident + 8)
        resident_keys, absent_keys = keys[:num_resident], keys[num_resident:]
        resident_slots = {
            torch.ops.fbgemm.lxu_cache_slot(k, cache_sets) for k in resident_keys
        }
        absent_keys.append(
            next(
                k
                for k in range(1000, 2000)
                if torch.ops.fbgemm.lxu_cache_slot(k, cache_sets) in resident_slots
            )
        )
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            source = (
                torch.randn(num_resident, embedding_dim, device=device) * 3 + 1
            ).to(dtype)
            source[0].fill_(2.5)
            resident = torch.tensor(resident_keys, dtype=torch.int64, device=device)

            int8_state = torch.full(
                (cache_sets, 1), -1, dtype=torch.int64, device=device
            )
            int8_weights = torch.zeros(
                (cache_sets, embedding_dim + 8), dtype=torch.uint8, device=device
            )
            torch.ops.fbgemm.kvzch_direct_cache_insert_int8(
                int8_state, int8_weights, resident, source
            )
            fp_state = torch.full_like(int8_state, -1)
            fp_weights = torch.zeros(
                (cache_sets, embedding_dim), dtype=dtype, device=device
            )
            torch.ops.fbgemm.kvzch_direct_cache_insert(
                fp_state, fp_weights, resident, source
            )
            torch.testing.assert_close(int8_state, fp_state, rtol=0, atol=0)

            src32 = source.float()
            row_min = src32.min(dim=1).values
            row_range = src32.max(dim=1).values - row_min
            expected_scale = torch.where(
                row_range == 0, torch.ones_like(row_range), row_range / 255
            )
            slots = [
                torch.ops.fbgemm.lxu_cache_slot(k, cache_sets) for k in resident_keys
            ]
            qparams = int8_weights[slots, embedding_dim:].clone().view(torch.float32)
            torch.testing.assert_close(qparams[:, 0], expected_scale)
            torch.testing.assert_close(qparams[:, 1], row_min, rtol=0, atol=0)

            lookup_keys = (resident_keys + absent_keys + resident_keys[:4]) * 3
            perm = torch.randperm(len(lookup_keys)).tolist()
            lookup_keys = [lookup_keys[i] for i in perm]
            indices = torch.tensor(lookup_keys, dtype=torch.int64, device=device)
            N = indices.numel()
            q_actions, q_count, q_inserted, q_addresses = self._run_direct_cache_lookup(
                torch.ops.fbgemm.kvzch_direct_cache_lookup_int8,
                indices,
                int8_state,
                int8_weights,
                embedding_dim,
                dtype,
            )
            f_actions, f_count, f_inserted, f_addresses = self._run_direct_cache_lookup(
                torch.ops.fbgemm.kvzch_direct_cache_lookup,
                indices,
                fp_state,
                fp_weights,
                embedding_dim,
                dtype,
            )
            count = int(q_count.item())
            self.assertEqual(count, int(f_count.item()))
            self.assertEqual(count, 3 * len(absent_keys))
            torch.testing.assert_close(q_actions[:count], f_actions[:count])
            self.assertTrue(set(q_actions[:count].tolist()) == set(absent_keys))

            miss_rows = torch.randn(count, embedding_dim, device=device).to(dtype)
            q_inserted[:count].copy_(miss_rows)
            f_inserted[:count].copy_(miss_rows)
            q_output = torch.empty((N, embedding_dim), dtype=dtype, device=device)
            torch.ops.fbgemm.kvzch_direct_cache_gather_int8(
                q_output, q_addresses, dtype
            )
            f_output = torch.empty_like(q_output)
            torch.ops.fbgemm.ssd_gather_from_row_addresses(f_output, f_addresses)

            is_hit = torch.tensor(
                [k in resident_keys for k in lookup_keys], device=device
            )
            torch.testing.assert_close((q_addresses & 1).bool(), is_hit, rtol=0, atol=0)
            torch.testing.assert_close(
                q_output[~is_hit], f_output[~is_hit], rtol=0, atol=0
            )
            source_pos = torch.tensor(
                [
                    resident_keys.index(k) if k in resident_keys else 0
                    for k in lookup_keys
                ],
                device=device,
            )
            expected = src32[source_pos]
            tolerance = (
                expected_scale[source_pos].unsqueeze(1) * 0.5 * (1 + 1e-4)
                + expected.abs() * torch.finfo(dtype).eps
                + 1e-6
            )
            error = (q_output.float() - expected).abs()
            self.assertTrue(bool((error[is_hit] <= tolerance[is_hit]).all()))
            torch.testing.assert_close(
                q_output[lookup_keys.index(resident_keys[0])].float(),
                src32[0],
                rtol=0,
                atol=0,
            )

    def test_kvzch_direct_cache_gather_int8_rejects_miss_dtype_mismatch(self) -> None:
        device = torch.accelerator.current_accelerator()
        row_dtype = torch.float16
        source = torch.tensor([[0.25, -0.5, 1.0, 3.0]], dtype=row_dtype, device=device)
        indices = torch.tensor([3], dtype=torch.int64, device=device)
        state = torch.full((2, 1), -1, dtype=torch.int64, device=device)
        weights = torch.empty((2, 12), dtype=torch.uint8, device=device)
        _, count, inserted, addresses = self._run_direct_cache_lookup(
            torch.ops.fbgemm.kvzch_direct_cache_lookup_int8,
            indices,
            state,
            weights,
            4,
            row_dtype,
        )
        self.assertEqual(count.item(), 1)
        inserted.copy_(source)
        wrong_output = torch.zeros((1, 4), dtype=torch.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "output.dtype == row_dtype"):
            torch.ops.fbgemm.kvzch_direct_cache_gather_int8(
                wrong_output, addresses, row_dtype
            )
        torch.testing.assert_close(wrong_output, torch.zeros_like(wrong_output))
        output = torch.empty_like(source)
        torch.ops.fbgemm.kvzch_direct_cache_gather_int8(output, addresses, row_dtype)
        torch.testing.assert_close(output, source, rtol=0, atol=0)

        meta_output = torch.empty((1, 4), dtype=torch.float32, device="meta")
        meta_addresses = torch.empty(1, dtype=torch.int64, device="meta")
        with self.assertRaisesRegex(RuntimeError, "output.dtype == row_dtype"):
            torch.ops.fbgemm.kvzch_direct_cache_gather_int8(
                meta_output, meta_addresses, row_dtype
            )

    def test_kvzch_direct_cache_gather_int8_float(self) -> None:
        device = torch.accelerator.current_accelerator()
        cache_sets = 4096
        for embedding_dim in (64, 128):
            resident_keys = self._unique_slot_keys(cache_sets, 2048)
            absent_keys = self._unique_slot_keys(cache_sets, 512, start=10**6)
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                source = (
                    torch.randn(len(resident_keys), embedding_dim, device=device) * 3
                    + 1
                ).to(dtype)
                int8_state = torch.full(
                    (cache_sets, 1), -1, dtype=torch.int64, device=device
                )
                int8_weights = torch.zeros(
                    (cache_sets, embedding_dim + 8), dtype=torch.uint8, device=device
                )
                torch.ops.fbgemm.kvzch_direct_cache_insert_int8(
                    int8_state,
                    int8_weights,
                    torch.tensor(resident_keys, dtype=torch.int64, device=device),
                    source,
                )
                pool = torch.tensor(
                    resident_keys + absent_keys, dtype=torch.int64, device=device
                )
                indices = pool[
                    torch.randint(pool.numel(), (20000,), device=device)
                ].contiguous()
                _, count, inserted, addresses = self._run_direct_cache_lookup(
                    torch.ops.fbgemm.kvzch_direct_cache_lookup_int8,
                    indices,
                    int8_state,
                    int8_weights,
                    embedding_dim,
                    dtype,
                )
                num_misses = int(count.item())
                self.assertGreater(num_misses, 0)
                inserted[:num_misses].copy_(
                    torch.randn(num_misses, embedding_dim, device=device) * 5
                )

                staged = torch.empty(
                    (indices.numel(), embedding_dim), dtype=dtype, device=device
                )
                torch.ops.fbgemm.kvzch_direct_cache_gather_int8(
                    staged, addresses, dtype
                )
                expected = staged.float()
                fused = torch.full_like(expected, float("nan"))
                torch.ops.fbgemm.kvzch_direct_cache_gather_int8_float(
                    fused, addresses, dtype
                )
                # Bit-identical to the gather + `.to(float32)` sequence.
                torch.testing.assert_close(
                    fused.view(torch.int32),
                    expected.view(torch.int32),
                    rtol=0,
                    atol=0,
                )

                # CPU reference. q * scale + bias is a single rounding (FMA) on
                # the GPU; float64 holds the 8-bit x 24-bit product exactly.
                is_hit = (addresses & 1).bool().cpu()
                slots = torch.tensor(
                    [
                        torch.ops.fbgemm.lxu_cache_slot(k, cache_sets)
                        for k in indices.tolist()
                    ]
                )
                cache_cpu = int8_weights.cpu()
                codes = cache_cpu[slots, :embedding_dim].double()
                qparams = cache_cpu[slots, embedding_dim:].clone().view(torch.float32)
                dequant = (
                    codes * qparams[:, :1].double() + qparams[:, 1:].double()
                ).float()
                reference = dequant.to(dtype).float()
                # Miss rows are read through their inserted-row address.
                miss_rows = (addresses.cpu()[~is_hit] - inserted.data_ptr()) // (
                    inserted.stride(0) * inserted.element_size()
                )
                reference[~is_hit] = inserted.cpu().float()[miss_rows]
                torch.testing.assert_close(
                    fused.cpu().view(torch.int32),
                    reference.view(torch.int32),
                    rtol=0,
                    atol=0,
                )

        empty = torch.empty((0, 128), dtype=torch.float32, device=device)
        torch.ops.fbgemm.kvzch_direct_cache_gather_int8_float(
            empty, torch.empty(0, dtype=torch.int64, device=device), torch.float16
        )
        with self.assertRaises(RuntimeError):
            torch.ops.fbgemm.kvzch_direct_cache_gather_int8_float(
                torch.empty((1, 128), dtype=torch.float16, device=device),
                torch.ones(1, dtype=torch.int64, device=device),
                torch.float16,
            )

    def test_kvzch_direct_cache_insert_int8_skips(self) -> None:
        device = torch.accelerator.current_accelerator()
        embedding_dim = 64
        cache_sets = 16
        keys = self._unique_slot_keys(cache_sets, 3)
        cache_state = torch.full((cache_sets, 1), -1, dtype=torch.int64, device=device)
        cache_weights = torch.zeros(
            (cache_sets, embedding_dim + 8), dtype=torch.uint8, device=device
        )
        rows = torch.randn(3, embedding_dim, device=device).half()
        rows[1].zero_()
        indices = torch.tensor(keys, dtype=torch.int64, device=device)

        torch.ops.fbgemm.kvzch_direct_cache_insert_int8(
            cache_state,
            cache_weights,
            indices,
            rows,
            torch.tensor([0], dtype=torch.int32, device=device),
        )
        self.assertTrue(bool((cache_state == -1).all()))
        self.assertTrue(bool((cache_weights == 0).all()))

        torch.ops.fbgemm.kvzch_direct_cache_insert_int8(
            cache_state, cache_weights, indices, rows
        )
        slots = [torch.ops.fbgemm.lxu_cache_slot(k, cache_sets) for k in keys]
        self.assertEqual(int(cache_state[slots[0]][0].item()), keys[0])
        self.assertEqual(int(cache_state[slots[1]][0].item()), -1)
        self.assertTrue(bool((cache_weights[slots[1]] == 0).all()))
        self.assertEqual(int(cache_state[slots[2]][0].item()), keys[2])

    def expand_tensor(
        self, tensor: torch.Tensor, size: int, max_val: int
    ) -> torch.Tensor:
        return torch.cat(
            [
                tensor,
                torch.randint(
                    low=0, high=max_val, size=(size - tensor.numel(),), dtype=torch.long
                ),
            ]
        )

    # pyre-ignore [56]
    @given(
        hash_size=st.integers(min_value=10, max_value=100),
        iters=st.integers(min_value=1, max_value=5),
        total_to_uniq_ratio=st.integers(min_value=1, max_value=5),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_scratch_pad_indices_queue(
        self,
        hash_size: int,
        iters: int,
        total_to_uniq_ratio: int,
    ) -> None:
        """
        Test correctness of
        `torch.classes.fbgemm.SSDScratchPadIndicesQueue` against the
        Python implementation.

        This test inserts a bunch of indices into
        `SSDScratchPadIndicesQueue` and then performs lookup
        """
        # Unique indices have to be between 0 and hash_size
        num_uniq_indices = random.randint(0, hash_size)
        num_uniq_lookup_indices = random.randint(0, hash_size)

        # The total number of indices can be arbitrary
        num_indices = num_uniq_indices * total_to_uniq_ratio
        num_lookup_indices = num_uniq_lookup_indices * total_to_uniq_ratio

        # Always -1 in SSD TBE
        sentinel_value = -1

        # Instantiate the SSDScratchPadIndicesQueue object
        sp_idx_queue = torch.classes.fbgemm.SSDScratchPadIndicesQueue(sentinel_value)

        all_indices = []
        all_counts = []
        all_lookup_indices = []
        all_lookup_counts = []
        for _ in range(iters):
            # Generate the scratch pad indices (conflict missed indices)
            indices = torch.randperm(hash_size, dtype=torch.long)
            masks = torch.randperm(hash_size, dtype=torch.long)[
                : max(num_uniq_indices // 5, 1)
            ]
            # Mark some indices to the sentinel value to indicate that
            # they are not missing indices
            indices[masks] = sentinel_value
            indices = self.expand_tensor(
                indices[:num_uniq_indices], size=num_indices, max_val=hash_size
            )

            # Generate the SSD indices (all missed indices)
            lookup_indices = self.expand_tensor(
                torch.randperm(hash_size)[:num_uniq_lookup_indices],
                size=num_lookup_indices,
                max_val=hash_size,
            )

            # Generate count for scratch pad indices
            count = torch.randint(
                low=1, high=num_uniq_indices + 1, size=(1,), dtype=torch.long
            )

            # Generate count for the SSD indices
            lookup_count = torch.randint(
                low=1, high=num_uniq_lookup_indices + 1, size=(1,), dtype=torch.long
            )

            # Insert scratch pad indices into the scratch pad index
            # queue
            sp_idx_queue.insert_cuda(indices, count)

            all_indices.append(indices)
            all_lookup_indices.append(lookup_indices)
            all_counts.append(count)
            all_lookup_counts.append(lookup_count)

        # Ensure that the insertions are done
        torch.cuda.synchronize()

        all_lookup_outputs = []
        all_lookup_outputs_ref = []
        for indices, lookup_indices, count, lookup_count in zip(
            all_indices, all_lookup_indices, all_counts, all_lookup_counts
        ):

            # Run reference
            # Prepare inputs for the reference run
            sp_prev_curr_map_ref = torch.zeros_like(lookup_indices)
            sp_curr_prev_map_ref = torch.empty_like(indices, dtype=torch.int).fill_(-1)
            sp_indices = indices.clone().tolist()
            ssd_indices = lookup_indices.clone().tolist()

            # Insert indices into the map
            sp_map = {}
            # pyrefly: ignore [bad-index]
            for i, idx in enumerate(sp_indices[: count.item()]):
                sp_map[idx] = i

            # Lookup
            # pyrefly: ignore [bad-argument-type]
            for i in range(lookup_count.item()):
                ssd_idx = ssd_indices[i]
                if ssd_idx in sp_map:
                    loc = sp_map[ssd_idx]
                    sp_prev_curr_map_ref[i] = loc
                    sp_curr_prev_map_ref[loc] = i
                    sp_indices[loc] = sentinel_value
                    ssd_indices[i] = sentinel_value
                else:
                    sp_prev_curr_map_ref[i] = sentinel_value

            all_lookup_outputs_ref.append(
                (
                    sp_prev_curr_map_ref,
                    torch.as_tensor(sp_indices),
                    torch.as_tensor(ssd_indices),
                )
            )

            # Run test
            sp_prev_curr_map = torch.zeros_like(lookup_indices)
            sp_curr_prev_map = torch.empty_like(indices, dtype=torch.int).fill_(-1)
            sp_idx_queue.lookup_mask_and_pop_front_cuda(
                sp_prev_curr_map,
                sp_curr_prev_map,
                indices,
                lookup_indices,
                lookup_count,
            )

            all_lookup_outputs.append((sp_prev_curr_map, indices, lookup_indices))

        # Ensure that the lookups are done
        torch.cuda.synchronize()

        # Compare results
        for test, ref in zip(all_lookup_outputs, all_lookup_outputs_ref):
            for name, test_, ref_ in zip(
                [
                    "scratch_pad_prev_curr_map",
                    "scratch_pad_curr_prev_map",
                    "scratch_pad_indices",
                    "ssd_indices",
                ],
                test,
                ref,
            ):
                assert torch.equal(test_, ref_), (
                    f"{name} fails: hash_size {hash_size}, "
                    f"num_uniq_indices {num_uniq_indices}, "
                    f"num_uniq_lookup_indices {num_uniq_lookup_indices}, "
                    f"iters {iters}, "
                    f"total_to_uniq_ratio {total_to_uniq_ratio}"
                )

    @given(
        num_indices=st.integers(min_value=1, max_value=128),
        num_tensors=st.integers(min_value=1, max_value=2),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_compact_indices(self, num_indices: int, num_tensors: int) -> None:
        """
        Test correctness of `torch.ops.fbgemm.compact_indices`
        """
        device = "cuda"
        index_types = [
            random.choice([torch.int, torch.long]) for _ in range(num_tensors)
        ]

        num_sentinels = random.randint(0, num_indices)
        count_val = random.randint(num_sentinels, num_indices)
        max_int32 = (2**31) - 1
        max_int64 = (2**63) - 1

        # Generate input indices
        indices = [
            torch.randint(
                low=0,
                high=max_int32 if dtype == torch.int else max_int64,
                size=(num_indices,),
                dtype=dtype,
                device=device,
            )
            for dtype in index_types
        ]
        # Generate count
        count = torch.as_tensor([count_val], dtype=torch.int, device=device)

        # Randomize the positions to be set to -1
        rand_pos = torch.randperm(count_val)[:num_sentinels]
        for idx in indices:
            idx[rand_pos] = -1

        # Allocate output indices
        compact_indices = [
            torch.empty(count_val - num_sentinels, dtype=dtype, device=device)
            for dtype in index_types
        ]

        # Generate masks
        masks = torch.where(indices[0] != -1, 1, 0)

        # Allocate compact count
        compact_count = torch.empty(1, dtype=torch.int, device=device)

        # Run test
        torch.ops.fbgemm.compact_indices(
            compact_indices,
            compact_count,
            indices,
            masks,
            count,
        )

        # Compute the reference compact count
        actual_masks = masks[:count_val]
        masked_pos = actual_masks == 1
        compact_masks = actual_masks[masked_pos]
        ref_compact_count = compact_masks.numel()

        # Compare compact counts
        assert (
            compact_count == ref_compact_count
        ), f"Mismatch compact counts ({compact_count} vs {ref_compact_count})"

        # Compare compact indices
        for t, (idx, comp_idx) in enumerate(zip(indices, compact_indices)):  # noqa B007
            comp_idx = comp_idx[:ref_compact_count]
            idx = idx[:count_val][masked_pos]
            assert torch.equal(
                idx, comp_idx
            ), "Mismatch compact indices for index tensor # {t}"


# pyrefly: ignore [bad-argument-type]
@unittest.skipIf(*running_in_oss)
# pyrefly: ignore [bad-argument-type]
@unittest.skipIf(*gpu_unavailable)
class DirectMappedEnrichmentCacheInt8Test(unittest.TestCase):
    @staticmethod
    def _cache(
        fused_float_gather: bool,
        output_dtype: torch.dtype,
        int8: bool = True,
        read_populate: bool = False,
    ):  # noqa: ANN205
        from fbgemm_gpu.tbe.ssd.training import _DirectMappedEnrichmentCache

        device = torch.accelerator.current_accelerator(True)
        assert device is not None
        return _DirectMappedEnrichmentCache(
            embedding_dim=128,
            total_hash_size=1 << 20,
            cache_dtype=torch.float16,
            device=device,
            cache_size_mb=1,
            read_populate=read_populate,
            read_populate_max_lookups=0,
            int8=int8,
            fused_float_gather=fused_float_gather,
            output_dtype=output_dtype,
        )

    @staticmethod
    def _backend(
        table: torch.Tensor,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]:
        def read_backend_into(
            action_indices_cpu: torch.Tensor,
            rows: torch.Tensor,
            count_cpu: torch.Tensor,
        ) -> None:
            torch.accelerator.synchronize()
            count = int(count_cpu[0])
            keys = action_indices_cpu[:count].to(table.device)
            rows[:count].copy_(table[keys])

        return read_backend_into

    def test_int8_lookup_matches_backend_rows(self) -> None:
        cache = self._cache(fused_float_gather=False, output_dtype=torch.float16)
        self.assertEqual(cache.cache_weights.dtype, torch.uint8)
        self.assertEqual(cache.cache_weights.shape[1], 128 + 8)
        device = torch.accelerator.current_accelerator(True)
        table = (torch.randn(4096, 128, device=device) * 3 + 1).half()
        resident = torch.arange(0, 4096, 2, device=device)
        cache.populate_from_write(resident, table[resident])
        indices = torch.randint(0, 4096, (50000,), device=device)
        output = cache.lookup(indices, self._backend(table))
        hits, queries = cache.last_hit_counts()
        self.assertGreater(hits, 0)
        self.assertLess(hits, queries)
        self.assertEqual(output.dtype, torch.float16)
        expected = table[indices].float()
        row_range = expected.amax(dim=1) - expected.amin(dim=1)
        max_err = (output.float() - expected).abs().amax(dim=1)
        self.assertTrue(bool((max_err <= row_range / 255 * 0.5 + 1e-2).all()))
        misses = (indices % 2) == 1
        torch.testing.assert_close(output[misses], table[indices[misses]])

    def test_fused_float_gather_is_bit_identical(self) -> None:
        cache = self._cache(fused_float_gather=True, output_dtype=torch.float32)
        self.assertTrue(cache._fused_float_gather)
        device = torch.accelerator.current_accelerator(True)
        table = (torch.randn(4096, 128, device=device) * 3 + 1).half()
        resident = torch.arange(0, 4096, 2, device=device)
        cache.populate_from_write(resident, table[resident])
        indices = torch.randint(0, 4096, (50000,), device=device)
        fused = cache.lookup(indices, self._backend(table))
        cache._fused_float_gather = False
        unfused = cache.lookup(indices, self._backend(table))
        self.assertEqual(fused.dtype, torch.float32)
        self.assertEqual(unfused.dtype, torch.float16)
        torch.testing.assert_close(
            fused.view(torch.int32),
            unfused.to(torch.float32).view(torch.int32),
            rtol=0,
            atol=0,
        )

    def test_lookup_read_population_for_fp_and_int8(self) -> None:
        device = torch.accelerator.current_accelerator(True)
        table = (torch.randn(4096, 128, device=device) * 3 + 1).half()
        for int8 in (False, True):
            with self.subTest(int8=int8):
                cache = self._cache(
                    fused_float_gather=False,
                    output_dtype=torch.float16,
                    int8=int8,
                    read_populate=True,
                )
                keys = SSDUtilsTest._unique_slot_keys(cache.cache_state.size(0), 32)
                indices = torch.tensor(keys, dtype=torch.int64, device=device)
                first = cache.lookup(indices, self._backend(table))
                self.assertEqual(cache.last_hit_counts(), (0, 32))
                torch.testing.assert_close(first, table[indices], rtol=0, atol=0)
                second = cache.lookup(indices, self._backend(table))
                self.assertEqual(cache.last_hit_counts(), (32, 32))
                if int8:
                    expected = table[indices].float()
                    row_range = expected.amax(dim=1) - expected.amin(dim=1)
                    error = (second.float() - expected).abs().amax(dim=1)
                    self.assertTrue(bool((error <= row_range / 510 + 1e-2).all()))
                else:
                    torch.testing.assert_close(second, first, rtol=0, atol=0)

    def test_fused_float_gather_requires_fp32_output(self) -> None:
        cache = self._cache(fused_float_gather=True, output_dtype=torch.bfloat16)
        self.assertFalse(cache._fused_float_gather)
