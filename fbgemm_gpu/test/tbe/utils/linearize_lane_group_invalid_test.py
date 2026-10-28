# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Route test for TBE_ROCM_LINEARIZE_GROUP: an invalid value falls back to the stock loop.

Only exactly "8", "16" or "32" selects the narrow loop; anything else must be ignored, including a value with
trailing characters such as "8x" (which std::atoi would read as 8). This module uses "8x", so so linearize_index_kernel runs its
stock whole-wave loop. The gate is read once per process, so this module sets the invalid value before the first
call, in its own process. The outputs must equal the reference exactly (the stock loop's results).
"""

import math
import os
import unittest

os.environ["TBE_ROCM_LINEARIZE_GROUP"] = "8x"

# Registers torch.ops.fbgemm when the module runs under plain pytest (OSS CI), not only under buck.
import fbgemm_gpu  # noqa: E402, F401
import torch  # noqa: E402


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


def check_transpose(tc: unittest.TestCase, seed: int, T: int, B: int) -> None:
    """transpose_embedding_input on T non-VBE tables x B bags (0-24 ids, every 37th bag 150 ids, int32) must equal
    the reference exactly: linear indices = hash_size_cumsum[t] + index, infos = (t << info_B_num_bits) | b, their
    stable sort by linear index, and the run-length encoding (only the defined prefix of `run`, up to num_runs).
    """
    g = torch.Generator().manual_seed(seed)
    hash_sizes = torch.randint(50, 5000, (T,), generator=g)
    lens = torch.randint(0, 25, (T * B,), generator=g)
    lens[::37] = 150
    offsets = torch.zeros(T * B + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lens, 0)
    t_of_id = torch.repeat_interleave(torch.arange(T), lens.view(T, B).sum(1))
    b_of_id = torch.repeat_interleave(torch.arange(T * B) % B, lens)
    indices = (torch.rand(int(lens.sum()), generator=g) * hash_sizes[t_of_id]).long()
    hsc = torch.zeros(T + 1, dtype=torch.int64)
    hsc[1:] = torch.cumsum(hash_sizes, 0)
    bits = int(math.log2(int(hsc[-1])) + 1)
    info_B_num_bits, info_B_mask = torch.ops.fbgemm.get_infos_metadata(hsc.cuda(), B, T)
    linear, linear_sorted, infos_sorted, run, run_lengths, num_runs, _ = (
        torch.ops.fbgemm.transpose_embedding_input(
            hsc.cuda(),
            bits,
            indices.int().cuda(),
            offsets.int().cuda(),
            info_B_num_bits=info_B_num_bits,
            info_B_mask=info_B_mask,
        )
    )
    linear_ref = (hsc[t_of_id] + indices).int()
    infos_ref = ((t_of_id << info_B_num_bits) | b_of_id).int()
    order = torch.sort(linear_ref.long(), stable=True).indices
    sorted_ref = linear_ref[order]
    run_ref, run_lengths_ref = torch.unique_consecutive(sorted_ref, return_counts=True)
    pairs = [
        ("linear indices", linear.cpu(), linear_ref),
        ("sorted linear indices", linear_sorted.cpu(), sorted_ref),
        ("sorted infos", infos_sorted.cpu().int(), infos_ref[order]),
    ]
    for what, got, want in pairs:
        tc.assertTrue(torch.equal(got, want), _diff(what, got, want))
    n = int(num_runs.cpu()[0])
    tc.assertEqual(n, run_ref.numel(), "num_runs")
    got_run, got_len = run.cpu()[:n], run_lengths.cpu()[:n].long()
    tc.assertTrue(torch.equal(got_run, run_ref), _diff("run", got_run, run_ref))
    tc.assertTrue(
        torch.equal(got_len, run_lengths_ref),
        _diff("run lengths", got_len, run_lengths_ref),
    )


@unittest.skipUnless(_rocm(), "ROCm only")
class LinearizeLaneGroupInvalidTest(unittest.TestCase):
    def test_invalid_value_keeps_stock_results(self) -> None:
        """TBE_ROCM_LINEARIZE_GROUP=8x (invalid): 6 tables x 4,096 bags give exactly the reference outputs."""
        check_transpose(self, seed=5, T=6, B=4096)


if __name__ == "__main__":
    unittest.main()
