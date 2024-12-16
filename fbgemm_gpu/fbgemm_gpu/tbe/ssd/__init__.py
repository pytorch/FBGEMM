#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Load the prelude
from .common import ASSOC  # noqa: F401

# Load the inference and training ops
from .inference import SSDIntNBitTableBatchedEmbeddingBags  # noqa: F401
from .training import SSDTableBatchedEmbeddingBags  # noqa: F401
