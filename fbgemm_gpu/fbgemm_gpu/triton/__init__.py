#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Attempt to import triton kernels, fallback to reference if we cannot.
from .common import RoundingMode  # noqa

try:
    from .quantize import (
        triton_dequantize_mx4 as dequantize_mx4,
        triton_quantize_mx4 as quantize_mx4,
    )
except ImportError:
    from .quantize_ref import (  # noqa: F401, E402
        py_dequantize_mx4 as dequantize_mx4,
        py_quantize_mx4 as quantize_mx4,
    )
