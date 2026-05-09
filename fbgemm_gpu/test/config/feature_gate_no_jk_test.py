#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

# pyre-fixme[21]
import fbgemm_gpu
from fbgemm_gpu.config import FeatureGate, FeatureGateName

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import TestSuite
else:
    from fbgemm_gpu.test.test_utils import TestSuite


# This test runs in a binary launched with the following environment
# (set by the BUCK target):
#
#   FBGEMM_NO_JK=2                       -> EnvFirstThenJk policy (FBCODE)
#   FBGEMM_TBE_V2=1                      -> feature is "enabled" via env
#   FBGEMM_BOUNDS_CHECK_INDICES_V2=0     -> feature is "set but disabled"
#
# Each assertion uses a distinct FeatureGateName because the underlying C++
# FeatureGate singleton's per-key cache is process-lifetime.
@unittest.skipIf(open_source, "FBGEMM_NO_JK is FBCODE-only behavior")
class FeatureGateNoJkTest(TestSuite):  # pyre-ignore[11]
    def test_env_set_to_one_returns_enabled(self) -> None:
        self.assertTrue(FeatureGate.is_enabled(FeatureGateName.TBE_V2))

    def test_env_set_to_zero_returns_disabled(self) -> None:
        self.assertFalse(
            FeatureGate.is_enabled(FeatureGateName.BOUNDS_CHECK_INDICES_V2)
        )


if __name__ == "__main__":
    unittest.main()
