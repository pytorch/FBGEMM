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
    out_path: str, threads: Optional[int], tables_per_thread: Optional[int]
) -> torch.Tensor:
    """Run the worker in a fresh process with the given threading env and load
    its forward output. The thread count is read once (cached) at the first
    kernel call, so each setting needs its own process."""
    worker = _WORKER
    assert worker is not None  # guaranteed by the skipUnless on the test class
    env = dict(os.environ)
    env.pop("TBE_TABLE_THREADS", None)
    env.pop("TBE_TABLES_PER_THREAD", None)
    if threads is not None:
        env["TBE_TABLE_THREADS"] = str(threads)
    if tables_per_thread is not None:
        env["TBE_TABLES_PER_THREAD"] = str(tables_per_thread)
    subprocess.run([worker, out_path], env=env, check=True)
    return torch.load(out_path)


@unittest.skipUnless(
    _WORKER is not None,
    "requires the fbcode worker binary via NBIT_THREADING_WORKER ($(location)); "
    "not available in the OSS build",
)
class NBitForwardThreadingTest(unittest.TestCase):
    def test_threading_does_not_change_result(self) -> None:
        # Each config maps to (TBE_TABLE_THREADS, TBE_TABLES_PER_THREAD).
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


if __name__ == "__main__":
    unittest.main()
