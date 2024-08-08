#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from contextlib import contextmanager

# pyre-fixme[21]
import fbgemm_gpu
from fbgemm_gpu.config import FeatureGateName

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if not open_source:
    # pyre-fixme[21]
    from fbgemm_gpu.fb.config import FeatureGateName as FbFeatureGateName


class FeatureGateTest(unittest.TestCase):
    @contextmanager
    # pyre-ignore[2]
    def assertNotRaised(self, exc_type) -> None:
        try:
            # pyre-ignore[7]
            yield None
        except exc_type as e:
            raise self.failureException(e)

    def test_feature_gates(self) -> None:
        for feature in FeatureGateName:
            # pyre-ignore[16]
            with self.assertNotRaised(Exception):
                print(f"\n[OSS] Feature {feature.name} enabled: {feature.is_enabled()}")

    @unittest.skipIf(open_source, "Not supported in open source")
    def test_feature_gates_fb(self) -> None:
        # pyre-fixme[16]
        for feature in FbFeatureGateName:
            # pyre-ignore[16]
            with self.assertNotRaised(Exception):
                print(f"\n[FB] Feature {feature.name} enabled: {feature.is_enabled()}")


if __name__ == "__main__":
    unittest.main()
