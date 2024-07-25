#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from contextlib import contextmanager

from fbgemm_gpu.config import FeatureGateName


class FeatureGateTest(unittest.TestCase):
    @contextmanager
    # pyre-ignore[2]
    def assertNotRaised(self, exc_type) -> None:
        try:
            # pyre-ignore[7]
            yield None
        except exc_type as e:
            raise self.failureException(e)

    def test_feature_gate(self) -> None:
        for feature in FeatureGateName:
            # pyre-ignore[16]
            with self.assertNotRaised(Exception):
                print(f"\nFeature {feature.name} enabled: {feature.is_enabled()}")


if __name__ == "__main__":
    unittest.main()
