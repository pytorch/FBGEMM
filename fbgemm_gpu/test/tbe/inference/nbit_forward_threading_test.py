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
    row_parallelism: Optional[bool] = None,
    bag_parallelism: Optional[bool] = None,
    bags_per_thread: Optional[int] = None,
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
    env.pop("FBGEMM_NO_JK", None)
    env.pop("FBGEMM_TBE_NOBAG_ROW_PARALLELISM", None)
    env.pop("FBGEMM_TBE_BAG_PARALLELISM", None)
    env.pop("FBGEMM_TBE_MIN_BAGS_PER_THREAD", None)
    env.pop("FBGEMM_TBE_BAG_PARALLELISM_MIN_B", None)
    # Stats collection forces the whole-table path, so it must not leak in from
    # the ambient environment and silently skip the scheduler under test.
    env.pop("FBGEMM_STATS_ENABLE", None)
    if threads is not None:
        env["FBGEMM_TBE_MAX_NUM_THREADS"] = str(threads)
    if tables_per_thread is not None:
        env["FBGEMM_TBE_MIN_TABLES_PER_THREAD"] = str(tables_per_thread)
    if rows_per_thread is not None:
        env["FBGEMM_TBE_MIN_ROWS_PER_THREAD"] = str(rows_per_thread)
    if row_parallelism is not None:
        # TBE_NOBAG_ROW_PARALLELISM resolves through JK in fbcode and through
        # an (absent, hence false) env var in OSS, so left alone it makes
        # coverage depend on live JK state. FBGEMM_NO_JK=1 switches the gate to
        # env-only lookup so the value below is what the worker actually gets;
        # the worker asserts that it took effect.
        env["FBGEMM_NO_JK"] = "1"
        env["FBGEMM_TBE_NOBAG_ROW_PARALLELISM"] = "1" if row_parallelism else "0"
    if bag_parallelism is not None:
        # Plain env var (no JK), read once per process like the thread count.
        env["FBGEMM_TBE_BAG_PARALLELISM"] = "1" if bag_parallelism else "0"
    if bags_per_thread is not None:
        env["FBGEMM_TBE_MIN_BAGS_PER_THREAD"] = str(bags_per_thread)
    subprocess.run([worker, out_path, mode], env=env, check=True)
    return torch.load(out_path)


def _ran_up_front_fill(out_path: str) -> bool:
    """Whether that pooled run did the up-front fill (skipped only when
    flattened)."""
    with open(out_path + ".fill") as f:
        return f.read() == "1"


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
            self.assertGreater(
                int(torch.count_nonzero(base)), 0, "reference output is all zeros"
            )
            for name, out in outputs.items():
                self.assertEqual(out.shape, base.shape, f"{name}: shape mismatch")
                self.assertTrue(
                    torch.equal(out, base),
                    f"{name} output differs from single_thread (threading changed the result)",
                )

    def _assert_bag_chunking_is_bitwise_identical(
        self, mode: str, flattened: bool = True
    ) -> None:
        # The POOLED path splits a table's BAG RANGE across threads and draws
        # (table, bag-range) pairs from one flat work list, so tables and bags
        # are balanced by a single dynamic schedule instead of a nested region
        # per table. The pooled_bags workloads have only 4 tables against 1023
        # bags, so table parallelism alone cannot use more than 4 threads and
        # the higher-thread arms are genuinely exercising the bag chunker.
        #
        # Every path gives the same bits, so each arm also asserts which path
        # ran. `flattened=False` (B=1, padded total_D): no arm may take it.
        #
        # Outputs must be BITWISE identical. A bag is never split -- its whole
        # pooling reduction happens inside one kernel call -- and chunks write
        # disjoint output rows, so no partial sums cross threads and there is no
        # floating-point reordering however the bags are partitioned.
        #
        # Config is (MAX_NUM_THREADS, MIN_BAGS_PER_THREAD, BAG_PARALLELISM,
        # expects flattened). `gate_off_8T` is the old whole-table path.
        configs = {
            "single_thread": (1, None, True, False),
            "default_no_env": (None, None, True, False),  # serial: cap is 1
            "gate_off_8T": (8, None, False, False),
            "2T_bags": (2, 1, True, True),
            "4T_bags": (4, 1, True, True),
            "8T_bags": (8, 1, True, True),
            # Grain small enough to split each table many ways.
            "8T_fine": (8, 8, True, True),
            # Bags-per-thread above the total bag count -> serial fallback even
            # though a thread cap is set.
            "8T_below_onset": (8, 10_000_000, True, False),
        }
        with tempfile.TemporaryDirectory() as d:
            outputs = {
                name: _run(
                    os.path.join(d, f"{mode}_{name}.pt"),
                    thr,
                    None,
                    mode=mode,
                    bag_parallelism=gate,
                    bags_per_thread=bpt,
                )
                for name, (thr, bpt, gate, _) in configs.items()
            }
            for name, (_, _, _, flat) in configs.items():
                self.assertEqual(
                    not _ran_up_front_fill(os.path.join(d, f"{mode}_{name}.pt")),
                    flat and flattened,
                    f"{mode}/{name}: expected the "
                    f"{'flattened' if flat and flattened else 'up-front-fill'} "
                    "path",
                )
            base = outputs["single_thread"]
            self.assertGreater(base.numel(), 0, f"{mode}: reference output is empty")
            self.assertGreater(
                int(torch.count_nonzero(base)),
                0,
                f"{mode}: reference output is all zeros",
            )
            self.assertTrue(
                torch.isfinite(base).all(), f"{mode}: reference output not finite"
            )
            for name, out in outputs.items():
                self.assertEqual(
                    out.shape, base.shape, f"{mode}/{name}: shape mismatch"
                )
                self.assertTrue(
                    torch.equal(out, base),
                    f"{mode}/{name} output differs from single_thread "
                    "(bag chunking changed the result)",
                )

    def test_bag_chunking_does_not_change_result(self) -> None:
        self._assert_bag_chunking_is_bitwise_identical("pooled_bags")

    def test_bag_chunking_mean_pooling_does_not_change_result(self) -> None:
        # MEAN normalises by each bag's own length, so a chunk given the wrong
        # bag count divides by the wrong divisor rather than just misplacing a row.
        self._assert_bag_chunking_is_bitwise_identical("pooled_bags_mean")

    def test_bag_chunking_weighted_does_not_change_result(self) -> None:
        # Per-sample weights are read from each chunk's rebased offset, so a
        # wrong offset multiplies the right rows by the wrong weights.
        self._assert_bag_chunking_is_bitwise_identical("pooled_bags_weighted")

    def test_bag_parallelism_b1_does_not_change_result(self) -> None:
        # B=1 is below FBGEMM_TBE_BAG_PARALLELISM_MIN_B (default 2), so with
        # bag parallelism on these lookups run serially rather than falling
        # back to table threading; every arm must still match serial.
        self._assert_bag_chunking_is_bitwise_identical("pooled_b1", flattened=False)

    def test_bag_chunking_padded_total_d_keeps_zero_fill(self) -> None:
        # total_D past D_offsets[T]: every arm must run the up-front fill. The
        # profiler assertion catches a skipped fill, whatever the allocator.
        self._assert_bag_chunking_is_bitwise_identical(
            "pooled_bags_padded_total_d", flattened=False
        )

    def test_bag_chunking_int64_offsets_does_not_change_result(self) -> None:
        # int64 indices/offsets hit the other generated kernel instantiation.
        self._assert_bag_chunking_is_bitwise_identical("pooled_bags_int64")

    def test_bag_chunking_fp32_output_does_not_change_result(self) -> None:
        # 4-byte output_t for the per-chunk memset.
        self._assert_bag_chunking_is_bitwise_identical("pooled_bags_fp32")

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
        # Config is (MAX_NUM_THREADS, MIN_TABLES_PER_THREAD, MIN_ROWS_PER_THREAD,
        # TBE_NOBAG_ROW_PARALLELISM). The gate is pinned explicitly on every arm
        # so the chunked scheduler is guaranteed to run rather than depending on
        # live JK state; `gate_off_8T` is the old whole-table path, which the
        # chunked arms must match bit for bit.
        configs = {
            "single_thread": (1, None, None, True),
            "default_no_env": (None, None, None, True),  # serial: cap is 1
            "gate_off_8T": (8, 1, None, False),
            "2T_rows": (2, 1, 1, True),
            "4T_rows": (4, 1, 1, True),
            "8T_rows": (8, 1, 1, True),
            # A grain so small that the dominant table is split many ways.
            "8T_fine": (8, 1, 1024, True),
            # Rows-per-thread above the total row count -> falls back to serial
            # even though a thread cap is set.
            "8T_below_onset": (8, 1, 10_000_000, True),
        }
        with tempfile.TemporaryDirectory() as d:
            outputs = {
                name: _run(
                    os.path.join(d, f"{mode}_{name}.pt"),
                    thr,
                    tpt,
                    rpt,
                    mode=mode,
                    row_parallelism=gate,
                )
                for name, (thr, tpt, rpt, gate) in configs.items()
            }
            base = outputs["single_thread"]
            self.assertGreater(base.numel(), 0, f"{mode}: reference output is empty")
            self.assertGreater(
                int(torch.count_nonzero(base)),
                0,
                f"{mode}: reference output is all zeros",
            )
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
