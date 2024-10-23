#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa: F401
# pyre-strict

# Load custom operator libraries and register shape functions.
from fbgemm_gpu.experimental.gen_ai.custom_ops import (
    attention_ops,
    comm_ops,
    gemm_ops,
    kv_cache_ops,
    quantize_ops,
)
