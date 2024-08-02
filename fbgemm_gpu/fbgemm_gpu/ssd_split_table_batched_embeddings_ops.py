#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import warnings

from fbgemm_gpu.tbe.ssd import (  # noqa: F401
    ASSOC,  # noqa: F401
    SSDIntNBitTableBatchedEmbeddingBags,  # noqa: F401
    SSDTableBatchedEmbeddingBags,  # noqa: F401
)


warnings.warn(  # noqa: B028
    f"""\033[93m
    The Python module {__name__} is now DEPRECATED and will be removed in the
    future.  Users should import fbgemm_gpu.tbe.ssd into their scripts instead.
    \033[0m""",
    DeprecationWarning,
)
