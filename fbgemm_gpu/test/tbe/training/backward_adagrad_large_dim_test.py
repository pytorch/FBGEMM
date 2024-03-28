#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import unittest
from typing import Any, Dict

from hypothesis import given, settings

from .backward_adagrad_common import (
    adjust_mixed_B_st,
    common_settings,
    common_strategy,
    execute_backward_adagrad,
    gpu_unavailable,
    optests,
    PoolingMode,
    skipIfRocm,
    SparseType,
    st,
)

# Set up test strategy
test_st: Dict[str, Any] = common_strategy.copy()
test_st["D"] = st.integers(min_value=128, max_value=512)


@optests.generate_opcheck_tests(fast=True)
class BackwardAdagradLargeDimTest(unittest.TestCase):
    @skipIfRocm("Unblock large dim enablement on other GPUs")
    @unittest.skipIf(*gpu_unavailable)
    @given(
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        mixed_B=st.booleans(),
        pooling_mode=st.sampled_from(PoolingMode),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_large_dims(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        """
        Test large embedding dimensions [512, 2048] with Adagrad optimizers
        """
        kwargs = adjust_mixed_B_st(kwargs)
        execute_backward_adagrad(
            compile=False,
            **kwargs,
        )


if __name__ == "__main__":
    unittest.main()
