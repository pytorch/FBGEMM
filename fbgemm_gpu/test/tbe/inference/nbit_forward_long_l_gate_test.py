#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NBitLongLNobagTest with the TBE_ROCM_INFERENCE_NOBAG_LONG_L gate on.

The gate is read once per process, so this module sets it before anything
reads it, in both OSS (env var) and fbcode (env var honored over the
JustKnob), and then runs the same tests. On gfx942 each case also asserts that
the long-L kernel ran exactly when the case is inside the specialization's
guard; on other GPUs the cases only check the output against the reference.
"""

import os
import unittest

os.environ.setdefault("FBGEMM_NO_JK", "1")
os.environ["FBGEMM_TBE_ROCM_INFERENCE_NOBAG_LONG_L"] = "1"

from . import nbit_forward_test  # noqa: E402


class NBitLongLNobagGateOnTest(nbit_forward_test.NBitLongLNobagTest):
    assert_long_l_dispatch = True


if __name__ == "__main__":
    unittest.main()
