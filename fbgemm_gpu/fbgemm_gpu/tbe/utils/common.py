# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import TypeVar

import torch

Deviceable = TypeVar(
    "Deviceable", torch.nn.EmbeddingBag, torch.nn.Embedding, torch.Tensor
)


def round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def get_device() -> torch.device:
    if torch.cuda.is_available():
        # pyre-fixme[7]: Expected `device` but got `Union[int, device]`.
        return torch.cuda.current_device()
    elif torch.mtia.is_available():
        # pyre-fixme[7]: Expected `device` but got `Union[int, device]`.
        return torch.mtia.current_device()
    else:
        return torch.device("cpu")


def to_device(t: Deviceable, use_cpu: bool) -> Deviceable:
    if use_cpu:
        # pyre-fixme[7]: Expected `Deviceable` but got `Union[Tensor, torch.nn.EmbeddingBag]`.
        return t.cpu()
    elif torch.cuda.is_available():
        # pyre-fixme[7]: Expected `Deviceable` but got `Union[Tensor, torch.nn.EmbeddingBag]`.
        return t.cuda()
    else:
        # pyre-fixme[7]: Expected `Deviceable` but got `Union[Tensor, torch.nn.EmbeddingBag]`.
        return t.to(device="mtia")
