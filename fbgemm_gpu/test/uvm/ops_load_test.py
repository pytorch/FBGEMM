#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest

import fbgemm_gpu
import hypothesis.strategies as st
import torch
from hypothesis import given, settings

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import cpu_and_maybe_gpu, TestSuite
else:
    from fbgemm_gpu.test.test_utils import cpu_and_maybe_gpu, TestSuite


class OpsLoadTest(TestSuite):  # pyre-ignore[11]
    @given(
        device=cpu_and_maybe_gpu(),
        host_mapped=st.booleans(),
    )
    @settings(deadline=None)
    def test_cpu_ops(self, device: torch.device, host_mapped: bool) -> None:
        with self.assertNotRaised(Exception):  # pyre-ignore[16]
            torch.ops.fbgemm.new_unified_tensor(
                torch.zeros(1, device=device, dtype=torch.float),
                [1000],
                host_mapped,
            )


if __name__ == "__main__":
    unittest.main()
