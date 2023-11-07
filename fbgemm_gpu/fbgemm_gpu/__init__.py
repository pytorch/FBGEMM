#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "fbgemm_gpu_py.so"))
except Exception as e:
    print(e)

# __init__.py is only used in OSS
# Use existence to check if fbgemm_gpu_py.so has already been loaded
open_source: bool = True

# Re-export docs
# Trigger meta registrations
from . import _fbgemm_gpu_docs, sparse_ops  # noqa: F401, E402  # noqa: F401, E402

# Re-export the version string from the auto-generated version file
from ._fbgemm_gpu_version import __version__  # noqa: F401, E402
