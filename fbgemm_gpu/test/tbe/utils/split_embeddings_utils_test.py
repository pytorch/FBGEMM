#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import math
import pickle
import random
import unittest
from itertools import accumulate
from typing import List, Tuple

import fbgemm_gpu  # noqa E402

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu import sparse_ops  # noqa: F401
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    PoolingMode,
)
from hypothesis import assume, given, HealthCheck, settings, Verbosity

from .. import common  # noqa E402
from ..common import MAX_EXAMPLES, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, use_cpu_strategy
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, use_cpu_strategy

from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import (
    generate_vbe_metadata,
)


VERBOSITY: Verbosity = Verbosity.verbose


def gen_inputs(
    hash_sizes: List[int],
    batch_size: int,
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    the lengths of bags are chosen from
    a uniform distribution from [0, max_len]
    """

    T = len(hash_sizes)

    offsets = [0]
    indices_per_table = []
    for t in range(T):
        len_sum = 0
        for _ in range(batch_size):
            length = random.randint(0, max_len)
            len_sum += length
            offsets.append(offsets[-1] + length)

        n_rows = hash_sizes[t]
        indices_per_table.append(torch.randint(n_rows, [len_sum], dtype=torch.int64))

    indices = torch.cat(indices_per_table, dim=0)
    offsets = torch.tensor(offsets, dtype=torch.int64)

    return indices, offsets


def transpose_embedding_input_ref(
    hash_size_cumsum: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    info_B_num_bits: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    reference implementation of torch.ops.fbgemm.transpose_embedding_input
    """

    T = hash_size_cumsum.numel() - 1
    B = (offsets.numel() - 1) // T
    linear_indices = torch.zeros_like(indices)
    infos = torch.zeros_like(indices)
    for b_t in range(B * T):
        t = b_t // B
        b = b_t % B
        start = int(offsets[b_t].item())
        end = int(offsets[b_t + 1].item())
        for i in range(start, end):
            linear_indices[i] = indices[i] + hash_size_cumsum[t]
            infos[i] = (t << info_B_num_bits) | b

    linear_indices_sorted, sorted_idx = torch.sort(linear_indices, stable=True)
    infos_sorted = infos[sorted_idx]

    (
        sorted_linear_indices_run,
        sorted_linear_indices_run_lengths,
    ) = torch.unique_consecutive(linear_indices_sorted, return_counts=True)

    sorted_linear_indices_num_runs = torch.tensor(
        sorted_linear_indices_run.numel(), dtype=torch.int64
    )
    sorted_linear_indices_cumulative_run_lengths = torch.tensor(
        [0] + list(accumulate(sorted_linear_indices_run_lengths.tolist())),
        dtype=torch.int64,
    )

    return (
        linear_indices,
        linear_indices_sorted,
        infos_sorted,
        sorted_linear_indices_run,
        sorted_linear_indices_run_lengths,
        sorted_linear_indices_num_runs,
        sorted_linear_indices_cumulative_run_lengths,
    )


class SplitEmbeddingsUtilsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(
        B=st.integers(min_value=10, max_value=25),
        T=st.integers(min_value=5, max_value=20),
        E=st.integers(min_value=10, max_value=50),
    )
    @settings(deadline=30000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_transpose(self, B: int, T: int, E: int) -> None:
        hash_sizes = [random.randint(E, 2 * E) for _ in range(T)]
        batch_size = B
        max_len = 3 * E

        total_hash_size_bits: int = int(math.log2(sum(hash_sizes)) + 1)
        hash_size_cumsum = torch.tensor(
            [0] + list(accumulate(hash_sizes)), dtype=torch.int64
        )

        indices, offsets = gen_inputs(hash_sizes, batch_size, max_len)
        hash_size_cumsum_cuda = hash_size_cumsum.cuda()

        info_B_num_bits, _ = torch.ops.fbgemm.get_infos_metadata(
            hash_size_cumsum_cuda, B, T
        )

        (
            linear_indices,
            linear_indices_sorted,
            infos_sorted,
            sorted_linear_indices_run,
            sorted_linear_indices_run_lengths,
            sorted_linear_indices_num_runs,
            sorted_linear_indices_cumulative_run_lengths,
        ) = torch.ops.fbgemm.transpose_embedding_input(
            hash_size_cumsum_cuda,
            total_hash_size_bits,
            indices.cuda(),
            offsets.cuda(),
            info_B_num_bits=info_B_num_bits,
        )

        (
            linear_indices_ref,
            linear_indices_sorted_ref,
            infos_sorted_ref,
            sorted_linear_indices_run_ref,
            sorted_linear_indices_run_lengths_ref,
            sorted_linear_indices_num_runs_ref,
            sorted_linear_indices_cumulative_run_lengths_ref,
        ) = transpose_embedding_input_ref(
            hash_size_cumsum, indices, offsets, info_B_num_bits
        )

        self.assertTrue(torch.equal(linear_indices.cpu(), linear_indices_ref))
        self.assertTrue(
            torch.equal(linear_indices_sorted.cpu(), linear_indices_sorted_ref)
        )
        infos_sorted = infos_sorted.cpu()
        self.assertTrue(torch.equal(infos_sorted, infos_sorted_ref.to(torch.int32)))

        # fbgemm impl has padding so we need slice
        num = sorted_linear_indices_run_ref.numel()
        self.assertTrue(
            torch.equal(
                sorted_linear_indices_run.cpu()[:num], sorted_linear_indices_run_ref
            )
        )
        self.assertTrue(
            torch.equal(
                sorted_linear_indices_run_lengths.cpu()[:num],
                sorted_linear_indices_run_lengths_ref,
            )
        )
        self.assertEqual(
            sorted_linear_indices_num_runs.item(),
            sorted_linear_indices_num_runs_ref.item(),
        )
        self.assertTrue(
            torch.equal(
                sorted_linear_indices_cumulative_run_lengths.cpu()[: num + 1],
                sorted_linear_indices_cumulative_run_lengths_ref,
            )
        )

    @given(
        T=st.integers(min_value=1, max_value=64),
        B=st.integers(min_value=1, max_value=64),
        max_L=st.integers(min_value=1, max_value=64),
        bounds_check_mode=st.sampled_from(
            [
                BoundsCheckMode.FATAL,
                BoundsCheckMode.WARNING,
                BoundsCheckMode.IGNORE,
            ]
        ),
        use_cpu=use_cpu_strategy(),
        weighted=st.booleans(),
        dtype=st.sampled_from(
            [
                torch.int64,
                torch.int32,
            ]
        ),
        mixed_B=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_bounds_check(  # noqa C901
        self,
        T: int,
        B: int,
        max_L: int,
        bounds_check_mode: BoundsCheckMode,
        use_cpu: bool,
        weighted: bool,
        dtype: torch.dtype,
        mixed_B: bool,
    ) -> None:
        rows_per_table = torch.tensor(
            np.random.randint(low=1, high=1000, size=(T,))
        ).long()
        if not mixed_B:
            Bs = [B] * T
        else:
            low = max(int(0.25 * B), 1)
            high = int(B)
            if low == high:
                Bs = [B] * T
            else:
                Bs = [np.random.randint(low=low, high=high) for _ in range(T)]
        B_offsets = [0] + list(accumulate(Bs))
        Ls = np.random.randint(low=0, high=max_L, size=(B_offsets[-1],))
        indices = [
            np.random.randint(
                low=0,
                high=rows_per_table[t],
                size=sum(Ls[B_offsets[t] : B_offsets[t + 1]]),
            )
            for t in range(T)
        ]
        indices = torch.tensor(np.concatenate(indices, axis=0)).to(dtype)
        weights = (
            torch.rand(indices.shape, dtype=torch.float, device=indices.device)
            if weighted
            else None
        )
        offsets = torch.tensor([0] + np.cumsum(Ls.flatten()).tolist()).to(dtype)
        warning = torch.tensor([0]).long()

        if mixed_B:
            B_offsets = torch.tensor(
                B_offsets, device="cpu" if use_cpu else "cuda", dtype=torch.int32
            )
            max_B = max(Bs)
            vbe_metadata = generate_vbe_metadata(
                offsets,
                [[b] for b in Bs],
                pooling_mode=PoolingMode.SUM,
                feature_dims_cpu=torch.tensor(
                    [-1] * T, device="cpu", dtype=torch.int64
                ),  # unused
                device=torch.device("cuda") if not use_cpu else torch.device("cpu"),
            )
            B_offsets = vbe_metadata.B_offsets
            assert isinstance(B_offsets, torch.Tensor), "B_offsets must be tensor"
            info_B_num_bits, info_B_mask = torch.ops.fbgemm.get_infos_metadata(
                vbe_metadata.B_offsets,  # unused tensor
                vbe_metadata.max_B,
                B_offsets.numel() - 1,  # T
            )
            row_output_offsets, b_t_map = torch.ops.fbgemm.generate_vbe_metadata(
                B_offsets,
                vbe_metadata.B_offsets_rank_per_feature,
                vbe_metadata.output_offsets_feature_rank,
                torch.tensor(
                    [-1] * (T + 1),
                    device=torch.device("cuda") if not use_cpu else torch.device("cpu"),
                    dtype=torch.int,
                ),  # unused D_offsets
                -1,  # unused max_D
                False,  # nobag
                vbe_metadata.max_B_feature_rank,
                info_B_num_bits,
                offsets.numel() - 1,  # total_B
            )
            vbe_args = {
                "B_offsets": B_offsets,
                "max_B": max_B,
                "b_t_map": b_t_map,
                "info_B_num_bits": info_B_num_bits,
                "info_B_mask": info_B_mask,
            }
        else:
            vbe_args = {}

        self.assertEqual(indices.numel(), np.sum(Ls).item())
        self.assertEqual(offsets[-1], np.sum(Ls).item())
        if not use_cpu:
            indices, offsets, rows_per_table, warning = (
                indices.cuda(),
                offsets.cuda(),
                rows_per_table.cuda(),
                warning.cuda(),
            )
            if weighted:
                # pyre-fixme[16]: `Optional` has no attribute `cuda`.
                weights = weights.cuda()
        indices_copy = indices.clone()
        offsets_copy = offsets.clone()
        torch.ops.fbgemm.bounds_check_indices(
            rows_per_table,
            indices,
            offsets,
            bounds_check_mode,
            warning,
            weights,
            **vbe_args,
        )
        # we don't modify when we are in-bounds.
        torch.testing.assert_close(indices_copy, indices)
        indices[:] = torch.iinfo(dtype).max
        if bounds_check_mode != BoundsCheckMode.FATAL:
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                bounds_check_mode,
                warning,
                weights,
                **vbe_args,
            )
            torch.testing.assert_close(indices, torch.zeros_like(indices))
            if bounds_check_mode == BoundsCheckMode.WARNING:
                self.assertEqual(warning.item(), indices.numel())
        else:
            if use_cpu and indices.numel():
                with self.assertRaises(RuntimeError):
                    torch.ops.fbgemm.bounds_check_indices(
                        rows_per_table,
                        indices,
                        offsets,
                        bounds_check_mode,
                        warning,
                        weights,
                        **vbe_args,
                    )
            # It would be nice to test the CUDA implementation of BoundsCheckMode==FATAL,
            # but the device assert kills the CUDA context and requires a process restart,
            # which is a bit inconvenient.

        # test offsets bound errors
        indices = indices_copy.clone()
        offsets = offsets_copy.clone()
        if offsets.numel() > 0:
            offsets[0] = -100
        if offsets.numel() > 1:
            offsets[-1] += 100
        if bounds_check_mode != BoundsCheckMode.FATAL:
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                bounds_check_mode,
                warning,
                weights,
                **vbe_args,
            )
            if offsets.numel() > 0:
                self.assertEqual(offsets[0].item(), 0)
            if offsets.numel() > 1:
                self.assertEqual(offsets[-1].item(), indices.numel())
            if bounds_check_mode == BoundsCheckMode.WARNING:
                # -1 because when we have 2 elements in offsets, we have only 1
                # warning for the pair.
                self.assertGreaterEqual(warning.item(), min(2, offsets.numel() - 1))
        else:
            if use_cpu and indices.numel():
                with self.assertRaises(RuntimeError):
                    torch.ops.fbgemm.bounds_check_indices(
                        rows_per_table,
                        indices,
                        offsets,
                        bounds_check_mode,
                        warning,
                        weights,
                    )

        # test offsets.size(0) ! = B * T + 1 case. Here we test with T >= 2 case.
        # If T == 1, we will always get the even division.
        # (does not apply to mixed_B = True)
        if not mixed_B and T >= 2:
            indices = indices_copy.clone()
            offsets = offsets_copy.clone()
            offsets = torch.cat(
                (
                    offsets,
                    torch.tensor(
                        [indices.numel()] * (T - 1),
                        dtype=offsets.dtype,
                        device=offsets.device,
                    ),
                ),
                dim=0,
            )
            with self.assertRaises(RuntimeError):
                torch.ops.fbgemm.bounds_check_indices(
                    rows_per_table,
                    indices,
                    offsets,
                    bounds_check_mode,
                    warning,
                    weights,
                )

        # test weights.size(0) != indices.size(0) case
        weights = torch.rand(
            (indices.size(0) + 1,), dtype=torch.float, device=indices.device
        )
        with self.assertRaises(RuntimeError):
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                bounds_check_mode,
                warning,
                weights,
                **vbe_args,
            )

    @given(
        T=st.integers(min_value=1, max_value=5),
        B=st.integers(min_value=1, max_value=8),
        L=st.integers(min_value=0, max_value=8),
        use_cpu=use_cpu_strategy(),
        use_cpu_hashtable=st.booleans(),
        use_array_for_index_remapping=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_pruning(
        self,
        T: int,
        B: int,
        L: int,
        use_cpu: bool,
        use_cpu_hashtable: bool,
        use_array_for_index_remapping: bool,
    ) -> None:
        E = int(1000)
        LOAD_FACTOR = 0.8
        pruning_ratio = 0.5

        capacities = [int(B * L / LOAD_FACTOR) + 1 for _ in range(T)]
        original_E = int(E / (1.0 - pruning_ratio))

        # Enforce the size of original_E/B/L to get the unique indices
        assume(original_E > B * L)

        current_device = "cpu" if use_cpu else torch.cuda.current_device()

        if use_cpu_hashtable:
            assume(use_cpu)

        indices = torch.randint(low=0, high=original_E, size=(T, B, L))
        for t in range(T):
            while (
                torch.unique(
                    indices[t], return_counts=False, return_inverse=False
                ).numel()
                != indices[t].numel()
            ):
                indices[t] = torch.randint(low=0, high=original_E, size=(B, L))

        indices = indices.view(-1).int()
        dense_indices = torch.randint(low=0, high=E, size=(T, B, L)).view(-1).int()
        offsets = torch.tensor([L * b_t for b_t in range(B * T + 1)]).int()

        # Initialize and insert Hashmap index remapping based data structure
        hash_table = torch.empty(
            (sum(capacities), 2),
            dtype=torch.int64,
        )
        hash_table[:, :] = -1
        hash_table_offsets = torch.tensor([0] + np.cumsum(capacities).tolist()).long()

        torch.ops.fbgemm.pruned_hashmap_insert(
            indices, dense_indices, offsets, hash_table, hash_table_offsets
        )

        if use_cpu_hashtable:
            # pyre-ignore[16]
            ht = torch.classes.fbgemm.PrunedMapCPU()
            ht.insert(indices, dense_indices, offsets, T)

        # Initialize and insert Array index remapping based data structure
        index_remappings_array = torch.tensor(
            [-1] * original_E * T,
            dtype=torch.int64,
            device=current_device,
        )
        index_remappings_array_offsets = torch.empty(
            T + 1,
            dtype=torch.int64,
            device=current_device,
        )
        index_remappings_array_offsets[0] = 0
        for t in range(T):
            indice_t = (indices.view(T, B, L))[t].long().view(-1).to(current_device)
            dense_indice_t = (
                (dense_indices.view(T, B, L))[t].long().view(-1).to(current_device)
            )
            selected_indices = torch.add(indice_t, t * original_E)[:E]
            index_remappings_array[selected_indices] = dense_indice_t
            index_remappings_array_offsets[t + 1] = (
                index_remappings_array_offsets[t] + original_E
            )

        # Move data when using device
        if not use_cpu:
            (
                indices,
                dense_indices,
                offsets,
                hash_table,
                hash_table_offsets,
                index_remappings_array,
                index_remappings_array_offsets,
            ) = (
                indices.to(current_device),
                dense_indices.to(current_device),
                offsets.to(current_device),
                hash_table.to(current_device),
                hash_table_offsets.to(current_device),
                index_remappings_array.to(current_device),
                index_remappings_array_offsets.to(current_device),
            )

        # Lookup
        if use_cpu_hashtable:
            dense_indices_ = ht.lookup(indices, offsets)
        elif not use_array_for_index_remapping:  # hashmap based pruning
            dense_indices_ = torch.ops.fbgemm.pruned_hashmap_lookup(
                indices, offsets, hash_table, hash_table_offsets
            )
        else:  # array based pruning
            dense_indices_ = torch.ops.fbgemm.pruned_array_lookup(
                indices,
                offsets,
                index_remappings_array,
                index_remappings_array_offsets,
            )

        # Validate the lookup result
        torch.testing.assert_close(dense_indices, dense_indices_)

        # For array based pruning, it will be out-of-boundary for arbitrarily
        # large indices. We will rely on bound checker to make sure indices
        # are within the boundary.
        if not use_array_for_index_remapping:
            # now, use a value that does not exist in the original set of indices
            # and so should be pruned out.
            indices[:] = np.iinfo(np.int32).max

            if use_cpu_hashtable:
                dense_indices_ = ht.lookup(indices, offsets)
            elif not use_array_for_index_remapping:  # hashmap based pruning
                dense_indices_ = torch.ops.fbgemm.pruned_hashmap_lookup(
                    indices, offsets, hash_table, hash_table_offsets
                )
            else:  # array based pruning
                dense_indices_ = torch.ops.fbgemm.pruned_array_lookup(
                    indices,
                    offsets,
                    index_remappings_array,
                    index_remappings_array_offsets,
                )
            torch.testing.assert_close(dense_indices.clone().fill_(-1), dense_indices_)

    def test_pickle(self) -> None:
        # pyre-ignore[16]
        tensor_queue = torch.classes.fbgemm.TensorQueue(torch.empty(0))
        pickled = pickle.dumps(tensor_queue)
        unpickled = pickle.loads(pickled)  # noqa: F841


if __name__ == "__main__":
    unittest.main()
