# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Route test for TBE_ROCM_BOUNDS_CHECK_ADAPTIVE_ALL: with the gate OFF nothing changes.

The gate is read once per process, so this case cannot share a process with bounds_check_adaptive_all_test.py
(which turns it on); this separate module makes sure it is unset before the first bounds check. With the gate off, a
non-VBE WARNING-mode bounds check must keep its stock launch (blockDim.x == 64, one wave64 per bag), and its results
must equal the CPU bounds check, exactly as before the change.
"""

import json
import os
import tempfile
import unittest

os.environ.pop("TBE_ROCM_BOUNDS_CHECK_ADAPTIVE_ALL", None)

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
class BoundsCheckAdaptiveAllOffTest(unittest.TestCase):
    def test_gate_off_keeps_stock_launch_and_results(self) -> None:
        """Gate unset, WARNING mode, out-of-range ids: blockDim.x == 64 on every launch, and indices / offsets equal
        to the CPU bounds check."""
        idx, off, rows = _inputs()
        idx_cpu, off_cpu = idx.clone(), off.clone()
        _check(rows, idx_cpu, off_cpu, 1)
        idx_gpu, off_gpu, rows_gpu = idx.cuda(), off.cuda(), rows.cuda()
        blocks = _launch_blocks(lambda: _check(rows_gpu, idx_gpu, off_gpu, 2))
        self.assertTrue(blocks, "bounds_check_indices_kernel_v2 not seen")
        self.assertTrue(
            all(b[0] == 64 for b in blocks),
            f"launch blocks {blocks}: gate off must keep blockDim.x == 64",
        )
        self.assertTrue(
            torch.equal(idx_gpu.cpu(), idx_cpu),
            _diff("indices vs CPU", idx_gpu.cpu(), idx_cpu),
        )
        self.assertTrue(
            torch.equal(off_gpu.cpu(), off), _diff("offsets", off_gpu.cpu(), off)
        )


if __name__ == "__main__":
    unittest.main()
