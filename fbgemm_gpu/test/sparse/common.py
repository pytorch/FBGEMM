#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import os
import unittest
from typing import Callable, Dict, List, Optional, Type

import fbgemm_gpu
import torch
from torch._utils_internal import get_file_path_2
from torch.testing._internal.optests import generate_opcheck_tests


def extend_test_class(
    klass: Type[unittest.TestCase],
    # e.g. "test_faketensor__test_cumsum": [unittest.expectedFailure]
    # Please avoid putting tests here, you should put operator-specific
    # skips and failures in deeplearning/fbgemm/fbgemm_gpu/test/failures_dict.json
    # pyre-ignore[24]: Generic type `Callable` expects 2 type parameters.
    additional_decorators: Optional[Dict[str, List[Callable]]] = None,
) -> None:
    failures_dict_path: str = get_file_path_2(
        "", os.path.dirname(__file__), "failures_dict.json"
    )

    # pyre-ignore[24]: Generic type `Callable` expects 2 type parameters.
    base_decorators: Dict[str, List[Callable]] = {
        "test_pt2_compliant_tag_fbgemm_jagged_dense_elementwise_add": [
            # This operator has been grandfathered in. We need to fix this test failure.
            unittest.expectedFailure,
        ],
        "test_pt2_compliant_tag_fbgemm_jagged_dense_elementwise_add_jagged_output": [
            # This operator has been grandfathered in. We need to fix this test failure.
            unittest.expectedFailure,
        ],
    }

    additional_decorators = additional_decorators or {}

    # Only generate tests for PyTorch 2.2+
    if (
        torch.__version__ >= "2.2.*"
        and hasattr(torch.library, "impl_abstract")
        and not hasattr(fbgemm_gpu, "open_source")
    ):
        generate_opcheck_tests(
            klass,
            ["fb", "fbgemm"],
            failures_dict_path,
            {**base_decorators, **additional_decorators},
            [
                "test_schema",
                "test_autograd_registration",
                "test_faketensor",
                "test_aot_dispatch_dynamic",
            ],
        )
