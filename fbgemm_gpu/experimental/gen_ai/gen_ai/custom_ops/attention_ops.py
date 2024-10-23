# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from .load_custom_library import load_custom_library

# Load all custom attention operators.
load_custom_library(
    "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:attention_ops"
)
