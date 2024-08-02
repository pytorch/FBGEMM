#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import torch

try:
    if torch.version.hip:
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu:ssd_split_table_batched_embeddings_hip"
        )
    else:
        try:
            torch.ops.load_library(
                "//deeplearning/fbgemm/fbgemm_gpu:ssd_split_table_batched_embeddings"
            )
        except OSError:
            # Keep for BC: will be deprecated soon.
            torch.ops.load_library(
                "//deeplearning/fbgemm/fbgemm_gpu/fb:ssd_split_table_batched_embeddings"
            )
except Exception:
    pass


ASSOC = 32
