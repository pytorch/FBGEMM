# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch


def expect_contiguous(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        return x.contiguous()
    else:
        return x
