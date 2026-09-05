# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import subprocess
import tempfile
import unittest
from typing import Optional

import torch

# Path to the worker binary, injected via `$(location ...)` in the BUCK env. This
# test relies on a sibling python_binary located through buck, so it only runs in
# the fbcode build; in the OSS (pytest/CMake) build the env var is absent and the
# test is skipped (use .get(), not [], so import never raises during collection).
_WORKER: Optional[str] = os.environ.get("NBIT_THREADING_WORKER")


def _run(
    out_path: str,
    threads: Optional[int],
    tables_per_thread: Optional[int],
    rows_per_thread: Optional[int] = None,
    mode: str = "pooled",
) -> torch.Tensor:
    """Run the worker in a fresh process with the given threading env and load
    its forward output. The thread count is read once (cached) at the first
    kernel call, so each setting needs its own process."""
    worker = _WORKER
    assert worker is not None  # guaranteed by the skipUnless on the test class
    env = dict(os.environ)
    env.pop("FBGEMM_TBE_MAX_NUM_THREADS", None)
    env.pop("FBGEMM_TBE_MIN_TABLES_PER_THREAD", None)
    env.pop("FBGEMM_TBE_MIN_ROWS_PER_THREAD", None)
    if threads is not None:
        env["FBGEMM_TBE_MAX_NUM_THREADS"] = str(threads)
    if tables_per_thread is not None:
        env["FBGEMM_TBE_MIN_TABLES_PER_THREAD"] = str(tables_per_thread)
    if rows_per_thread is not None:
        env["FBGEMM_TBE_MIN_ROWS_PER_THREAD"] = str(rows_per_thread)
    subprocess.run([worker, out_path, mode], env=env, check=True)
    return torch.load(out_path)


@unittest.skipUnless(
    _WORKER is not None,
    "requires the fbcode worker binary via NBIT_THREADING_WORKER ($(location)); "
    "not available in the OSS build",
)
class NBitForwardThreadingTest(unittest.TestCase):
    def test_threading_does_not_change_result(self) -> None:
        # Each config maps to (FBGEMM_TBE_MAX_NUM_THREADS, FBGEMM_TBE_MIN_TABLES_PER_THREAD).
        # Outputs must be BITWISE identical across all of them: table-threading
        # partitions independent per-table work into disjoint output slices, with
        # no cross-thread reduction, so there is no floating-point reordering.
        configs = {
            "single_thread": (1, None),  # explicit serial
            "default_no_env": (None, None),  # no env var -> serial path
            "2T_guard": (2, None),  # 2 threads, default granularity (G=16)
            "2T_all": (2, 1),  # 2 threads, thread every call
            "4T_all": (4, 1),  # 4 threads, thread every call
        }
        with tempfile.TemporaryDirectory() as d:
            outputs = {
                name: _run(os.path.join(d, f"{name}.pt"), thr, tpt)
                for name, (thr, tpt) in configs.items()
            }
            base = outputs["single_thread"]
            self.assertTrue(torch.isfinite(base).all(), "reference output not finite")
            for name, out in outputs.items():
                self.assertEqual(out.shape, base.shape, f"{name}: shape mismatch")
                self.assertTrue(
                    torch.equal(out, base),
                    f"{name} output differs from single_thread (threading changed the result)",
                )

    def _assert_row_chunking_is_bitwise_identical(self, mode: str) -> None:
        # NOBAG splits a table's ROW RANGE across threads, not just whole
        # tables. The workload is deliberately skewed (one table holds ~85% of
        # the lookups) so table-level parallelism is nearly useless and the
        # chunked scheduler is genuinely exercised.
        #
        # Outputs must still be BITWISE identical: in NOBAG each output row is
        # an independent gather (memcpy of one weight row) writing a disjoint
        # output slice, so there is no cross-thread reduction and no
        # floating-point reordering, regardless of how the rows are partitioned.
        #
        # Config is (MAX_NUM_THREADS, MIN_TABLES_PER_THREAD, MIN_ROWS_PER_THREAD).
        configs = {
            "single_thread": (1, None, None),
            "default_no_env": (None, None, None),  # serial: cap defaults to 1
            "2T_rows": (2, 1, 1),
            "4T_rows": (4, 1, 1),
            "8T_rows": (8, 1, 1),
            # A grain so small that the dominant table is split many ways.
            "8T_fine": (8, 1, 1024),
            # Rows-per-thread above the total row count -> falls back to serial
            # even though a thread cap is set.
            "8T_below_onset": (8, 1, 10_000_000),
        }
        with tempfile.TemporaryDirectory() as d:
            outputs = {
                name: _run(
                    os.path.join(d, f"{mode}_{name}.pt"),
                    thr,
                    tpt,
                    rpt,
                    mode=mode,
                )
                for name, (thr, tpt, rpt) in configs.items()
            }
            base = outputs["single_thread"]
            self.assertGreater(base.numel(), 0, f"{mode}: reference output is empty")
            for name, out in outputs.items():
                self.assertEqual(
                    out.shape, base.shape, f"{mode}/{name}: shape mismatch"
                )
                self.assertTrue(
                    torch.equal(out, base),
                    f"{mode}/{name} output differs from single_thread "
                    "(row chunking changed the result)",
                )

    def test_row_chunking_does_not_change_result(self) -> None:
        # INT4 output: the NOBAG fast path ignores offsets entirely.
        self._assert_row_chunking_is_bitwise_identical("nobag_skewed")

    def test_row_chunking_fp16_output_does_not_change_result(self) -> None:
        # FP16 output consumes the synthesised unit-stride chunk offsets, so
        # this exercises the offset/index/output rebasing the INT4 path skips.
        self._assert_row_chunking_is_bitwise_identical("nobag_fp16")

    def test_row_chunking_int64_offsets_does_not_change_result(self) -> None:
        # int64 indices/offsets hit the other generated kernel instantiation.
        self._assert_row_chunking_is_bitwise_identical("nobag_fp16_int64")


if __name__ == "__main__":
    unittest.main()
