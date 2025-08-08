#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch


lib = torch.library.Library("fbgemm.examples", "FRAGMENT")
lib.define(
    """add_tensors_float(
            Tensor x,
            Tensor y
        ) -> Tensor
        """
)
lib.impl("add_tensors_float", torch.ops.fbgemm.add_tensors_float, "CPU")


def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.ops.fbgemm.add_tensors_float(a, b)


def add_tensors_nested_ns(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.ops.fbgemm.examples.add_tensors_float(a, b)


def sgemm(
    alpha: float, TA: torch.Tensor, TB: torch.Tensor, beta: float, TC: torch.Tensor
) -> torch.Tensor:
    return torch.ops.fbgemm.sgemm_float(alpha, TA, TB, beta, TC)
