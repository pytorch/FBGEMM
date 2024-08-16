#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Trigger the manual addition of docstrings to pybind11-generated operators
try:
    from . import jagged_tensor_ops  # noqa: F401
except Exception:
    pass
