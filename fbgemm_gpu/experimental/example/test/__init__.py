# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch


gpu_unavailable: tuple[bool, str] = (
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    "CUDA is not available or no GPUs detected",
)
