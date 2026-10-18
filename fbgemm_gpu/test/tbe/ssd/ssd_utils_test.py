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
