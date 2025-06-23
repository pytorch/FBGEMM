#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

from typing import Optional

import torch


def load_torch_module(
    unified_path: str,
    cuda_path: Optional[str] = None,
    hip_path: Optional[str] = None,
    mtia_path: Optional[str] = None,
) -> None:
    try:
        torch.ops.load_library(unified_path)
    except Exception:
        if torch.version.hip:
            if not hip_path:
                hip_path = f"{unified_path}_hip"
            torch.ops.load_library(hip_path)
        else:
            try:
                # pyre-ignore-next-line[21]
                import mtia.host_runtime.torch_mtia.dynamic_library  # noqa

                if mtia_path is not None:
                    torch.ops.load_library(mtia_path)
            except OSError:
                if not cuda_path:
                    cuda_path = f"{unified_path}_cuda"
                torch.ops.load_library(cuda_path)


def load_torch_module_bc(new_path: str, old_path: str) -> None:
    try:
        torch.ops.load_library(new_path)
    except Exception:
        torch.ops.load_library(old_path)
