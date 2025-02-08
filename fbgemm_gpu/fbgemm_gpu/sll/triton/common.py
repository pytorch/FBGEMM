# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch


def next_power_of_two(N: int) -> int:
    if N > 4096:
        raise Exception(f"{N} is too large that is not supported yet")

    if N > 2048:
        return 4096
    elif N > 1024:
        return 2048
    elif N > 512:
        return 1024
    elif N > 256:
        return 512
    elif N > 128:
        return 256
    elif N > 64:
        return 128
    elif N > 32:
        return 64
    else:
        return 32


def expect_contiguous(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        return x.contiguous()
    else:
        return x
