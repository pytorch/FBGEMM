# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for TBE_ROCM_BOUNDS_CHECK_ADAPTIVE_ALL=1: the ROCm adaptive bounds-check launch for non-VBE TBEs.

In WARNING / IGNORE mode, bounds_check_indices_kernel_v2 on ROCm already picks a narrow thread group per bag (1-32
lanes by mean bag length) for VBE TBEs; non-VBE TBEs use one wave64 per bag. The gate lets non-VBE TBEs take the
adaptive launch too. FATAL mode is not affected.

The gate is read once per process, so this module sets it before the first bounds check and every test here runs
with it ON. The gate-OFF case is bounds_check_adaptive_all_off_test.py, a separate module and process.

Input: 4 non-VBE tables x 65,536 bags of 0-24 ids (every 97th bag 300 ids), int32, with about 1e-4 of the ids out of
range. The CPU bounds check on the same input is the reference: it is a plain loop, independent of any launch shape.
"""

import json
import os
import tempfile
import unittest

os.environ["TBE_ROCM_BOUNDS_CHECK_ADAPTIVE_ALL"] = "1"

import torch  # noqa: E402
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (  # noqa: E402
    BoundsCheckMode,
)

ROWS = [5000, 3000, 800, 20000]
B = 65536


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


def _inputs(
    valid_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """4 tables x 65,536 bags of 0-24 ids (every 97th bag 300 ids), int32 ids and offsets, rows per table. Unless
    valid_only, about 1e-4 of the ids are out of range (never negative)."""
    g = torch.Generator().manual_seed(11)
    lens, ids = [], []
    for r in ROWS:
        L = torch.randint(0, 25, (B,), generator=g)
        L[::97] = 300
        x = torch.randint(0, r, (int(L.sum()),), generator=g)
        bad = torch.rand(x.numel(), generator=g) < 1e-4
        if not valid_only:
            x[bad] = r + torch.randint(0, 1000, (int(bad.sum()),), generator=g)
        lens.append(L)
        ids.append(x)
    lengths = torch.cat(lens)
    offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, 0)
    return torch.cat(ids).int(), offsets.int(), torch.tensor(ROWS, dtype=torch.int64)


def _check(
    rows: torch.Tensor,
    idx: torch.Tensor,
    off: torch.Tensor,
    version: int,
    mode: BoundsCheckMode = BoundsCheckMode.WARNING,
) -> torch.Tensor:
    warning = torch.zeros(1, dtype=torch.int64, device=idx.device)
    torch.ops.fbgemm.bounds_check_indices(
        rows, idx, off, mode.value, warning, bounds_check_version=version
    )
    return warning


def _expected(idx: torch.Tensor, off: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Every out-of-range id becomes 0; nothing else changes."""
    table = torch.repeat_interleave(
        torch.arange(len(ROWS)), torch.diff(off.long()).view(len(ROWS), B).sum(1)
    )
    expected = idx.clone()
    expected[idx.long() >= rows[table]] = 0
    return expected


def _launch_blocks(fn) -> list[list[int]]:
    """Run fn under the profiler and return the block dims of every bounds_check_indices_kernel_v2 launch."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.json")
        p.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f).get("traceEvents", [])
    blocks = [
        e.get("args", {}).get("block")
        for e in events
        if "bounds_check_indices_kernel_v2" in e.get("name", "")
    ]
    return [b for b in blocks if b]


@unittest.skipUnless(_rocm(), "ROCm only")
class BoundsCheckAdaptiveAllTest(unittest.TestCase):
    def test_matches_cpu_and_expected_corrections(self) -> None:
        """WARNING mode with out-of-range ids: the GPU bounds check must set exactly the out-of-range ids to 0, leave
        every other index and every offset unchanged, and agree bit for bit with the CPU bounds check.
        """
        idx, off, rows = _inputs()
        expected = _expected(idx, off, rows)
        self.assertGreater(
            int((expected != idx).sum()), 0, "the input must contain out-of-range ids"
        )
        idx_cpu, off_cpu = idx.clone(), off.clone()
        _check(rows, idx_cpu, off_cpu, 1)
        idx_gpu, off_gpu = idx.cuda(), off.cuda()
        _check(rows.cuda(), idx_gpu, off_gpu, 2)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(idx_gpu.cpu(), expected),
            _diff("indices vs expected", idx_gpu.cpu(), expected),
        )
        self.assertTrue(
            torch.equal(idx_gpu.cpu(), idx_cpu),
            _diff("indices vs CPU", idx_gpu.cpu(), idx_cpu),
        )
        self.assertTrue(
            torch.equal(off_gpu.cpu(), off), _diff("offsets (GPU)", off_gpu.cpu(), off)
        )
        self.assertTrue(torch.equal(off_cpu, off), _diff("offsets (CPU)", off_cpu, off))

    def test_launch_uses_narrow_groups(self) -> None:
        """WARNING mode: every bounds_check_indices_kernel_v2 launch must use blockDim.x < 64, the adaptive narrow
        group. This proves the gated path ran: without the gate (or on libraries without this change) a non-VBE launch
        is [64, 4, 1] and this fails."""
        idx, off, rows = _inputs()
        idx_gpu, off_gpu, rows_gpu = idx.cuda(), off.cuda(), rows.cuda()
        _check(rows_gpu, idx_gpu.clone(), off_gpu.clone(), 2)
        blocks = _launch_blocks(
            lambda: _check(rows_gpu, idx_gpu.clone(), off_gpu.clone(), 2)
        )
        self.assertTrue(blocks, "bounds_check_indices_kernel_v2 not seen")
        self.assertTrue(
            all(b[0] < 64 for b in blocks),
            f"launch blocks {blocks}: expected blockDim.x < 64",
        )

    def test_fatal_mode_is_untouched(self) -> None:
        """FATAL mode with the gate ON and an all-valid input: the launch must stay one wave64 per bag
        (blockDim.x == 64), because the gate only covers WARNING / IGNORE, and the indices must be unchanged.
        """
        idx, off, rows = _inputs(valid_only=True)
        idx_gpu, off_gpu, rows_gpu = idx.cuda(), off.cuda(), rows.cuda()
        blocks = _launch_blocks(
            lambda: _check(rows_gpu, idx_gpu, off_gpu, 2, BoundsCheckMode.FATAL)
        )
        self.assertTrue(blocks, "bounds_check_indices_kernel_v2 not seen")
        self.assertTrue(
            all(b[0] == 64 for b in blocks),
            f"launch blocks {blocks}: FATAL must keep blockDim.x == 64",
        )
        self.assertTrue(
            torch.equal(idx_gpu.cpu(), idx),
            _diff("indices (FATAL, valid input)", idx_gpu.cpu(), idx),
        )


if __name__ == "__main__":
    unittest.main()
