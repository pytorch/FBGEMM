# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for TBE_ROCM_SORT_RADIX_BITS=7: 7-bit onesweep passes for the ROCm linear-index sort.

With the gate, the radix sort in transpose_embedding_input uses 7-bit onesweep passes (1024 threads x 12 items, match
rank) instead of rocprim's tuned 8-bit ones, but only for 4-byte keys and values and only when 7-bit digits need no
more passes than 8-bit ones; everything else keeps the stock config. LSD radix sort is stable, so the output is the
same for any digit width.

The gate is read once per process, so this module sets it before the first call. The eligibility rule is decided per
call, so the eligible and ineligible cases share this process. Each case checks (1) which sort config ran, from the
onesweep kernel names, and (2) exact equality with the reference: linear indices, their stable sort, the sorted infos
and the run-length encoding (only the defined prefix of `run`, which is uninitialised past num_runs).
"""

import math
import os
import re
import unittest

os.environ["TBE_ROCM_SORT_RADIX_BITS"] = "7"

# Registers torch.ops.fbgemm when the module runs under plain pytest (OSS CI), not only under buck.
import fbgemm_gpu  # noqa: E402, F401
import torch  # noqa: E402

SEVEN_BIT = re.compile(r"kernel_config<1024u, 12u[^>]*>, 7u")


def _rocm() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


def _diff(what: str, got: torch.Tensor, want: torch.Tensor) -> str:
    """Assertion message: the number of differing elements and the first one (or the shapes, if they differ)."""
    if got.shape != want.shape:
        return f"{what}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    bad = (got != want).nonzero()
    if bad.numel() == 0:
        return f"{what}: equal"
    i = tuple(bad[0].tolist())
    return (
        f"{what}: {bad.shape[0]} of {want.numel()} elements differ; first at {i}: "
        f"got {got[i].item()}, want {want[i].item()}"
    )


def run_case(
    tc: unittest.TestCase,
    rows: int,
    T: int,
    B: int,
    index_dtype: torch.dtype,
    seed: int,
) -> tuple[int, list[str]]:
    """T tables of `rows` rows, B bags of 0-24 ids each; returns (key bits, onesweep kernel names) after checking
    every output against the reference."""
    g = torch.Generator().manual_seed(seed)
    hash_sizes = torch.full((T,), rows)
    lens = torch.randint(0, 25, (T * B,), generator=g)
    offsets = torch.zeros(T * B + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lens, 0)
    t_of_id = torch.repeat_interleave(torch.arange(T), lens.view(T, B).sum(1))
    b_of_id = torch.repeat_interleave(torch.arange(T * B) % B, lens)
    indices = (
        torch.rand(int(lens.sum()), generator=g, dtype=torch.float64) * rows
    ).long()
    hsc = torch.zeros(T + 1, dtype=torch.int64)
    hsc[1:] = torch.cumsum(hash_sizes, 0)
    bits = int(math.log2(int(hsc[-1])) + 1)
    info_B_num_bits, info_B_mask = torch.ops.fbgemm.get_infos_metadata(hsc.cuda(), B, T)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        linear, linear_sorted, infos_sorted, run, run_lengths, num_runs, _ = (
            torch.ops.fbgemm.transpose_embedding_input(
                hsc.cuda(),
                bits,
                indices.to(index_dtype).cuda(),
                offsets.to(index_dtype).cuda(),
                info_B_num_bits=info_B_num_bits,
                info_B_mask=info_B_mask,
            )
        )
        torch.cuda.synchronize()
    names = [e.key for e in p.key_averages() if "radix_sort_onesweep" in e.key]
    linear_ref = (hsc[t_of_id] + indices).to(linear.dtype)
    infos_ref = ((t_of_id << info_B_num_bits) | b_of_id).int()
    order = torch.sort(linear_ref.long(), stable=True).indices
    sorted_ref = linear_ref[order]
    run_ref, run_lengths_ref = torch.unique_consecutive(sorted_ref, return_counts=True)
    where = f"rows={rows} T={T} B={B} {index_dtype}"
    pairs = [
        ("linear indices", linear.cpu(), linear_ref),
        ("sorted linear indices", linear_sorted.cpu(), sorted_ref),
        ("sorted infos", infos_sorted.cpu().int(), infos_ref[order]),
    ]
    for what, got, want in pairs:
        tc.assertTrue(torch.equal(got, want), _diff(f"{where}: {what}", got, want))
    n = int(num_runs.cpu()[0])
    tc.assertEqual(n, run_ref.numel(), f"{where}: num_runs")
    got_run, got_len = run.cpu()[:n], run_lengths.cpu()[:n].long()
    tc.assertTrue(
        torch.equal(got_run, run_ref), _diff(f"{where}: run", got_run, run_ref)
    )
    tc.assertTrue(
        torch.equal(got_len, run_lengths_ref),
        _diff(f"{where}: run lengths", got_len, run_lengths_ref),
    )
    return bits, names


@unittest.skipUnless(_rocm(), "ROCm only")
class SortRadix7Test(unittest.TestCase):
    def test_eligible_keys_use_7_bit_passes(self) -> None:
        """6 tables x 300,000 rows (21-bit keys: three 7-bit passes, as many as 8-bit), int32: every onesweep kernel
        carries the 7-bit config (this fails on libraries without the change), and the outputs are exact.
        """
        bits, names = run_case(
            self, rows=300000, T=6, B=65536, index_dtype=torch.int32, seed=3
        )
        self.assertEqual(
            (bits + 6) // 7, (bits + 7) // 8, f"{bits}-bit keys should be eligible"
        )
        self.assertTrue(names, "no onesweep kernels seen")
        self.assertTrue(
            all(SEVEN_BIT.search(n) for n in names),
            f"expected the 7-bit config, saw {names[:1]}",
        )

    def test_ineligible_bit_width_keeps_stock_config(self) -> None:
        """6 tables x 600,000 rows (22-bit keys: 7-bit digits would need four passes, 8-bit three): the stock config
        must run, and the outputs are exact."""
        bits, names = run_case(
            self, rows=600000, T=6, B=16384, index_dtype=torch.int32, seed=4
        )
        self.assertGreater(
            (bits + 6) // 7, (bits + 7) // 8, f"{bits}-bit keys should be ineligible"
        )
        self.assertTrue(names, "no onesweep kernels seen")
        self.assertFalse(
            any(SEVEN_BIT.search(n) for n in names),
            f"unexpected 7-bit config in {names[:1]}",
        )

    def test_int64_indices_keep_stock_config(self) -> None:
        """Eligible 21-bit keys but int64 indices: the 7-bit config is sized for 4-byte keys and values, so the stock
        config must run, and the outputs are exact."""
        _, names = run_case(
            self, rows=300000, T=6, B=16384, index_dtype=torch.int64, seed=5
        )
        self.assertFalse(
            any(SEVEN_BIT.search(n) for n in names),
            f"unexpected 7-bit config in {names[:1]}",
        )

    def test_small_input_matches_reference(self) -> None:
        """2 tables x 1,000 rows x 8 bags: a few dozen ids, far below the 400,000-item limit under which the sort
        does not use onesweep. The outputs are exact whatever sort path runs."""
        run_case(self, rows=1000, T=2, B=8, index_dtype=torch.int32, seed=6)


if __name__ == "__main__":
    unittest.main()
